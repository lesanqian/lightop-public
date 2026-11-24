# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import List, Optional, Union
import glob
import importlib.util
import runpy
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, ROCM_HOME
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from packaging import version
from get_version import get_version, get_dtk_version

def _get_pytorch_version_int():
    version=torch.__version__.split('.') # '2.9.0'
    return int(version[0])*100+int(version[1])*10+int(version[2])

dtk_version = get_dtk_version(ROCM_HOME)
torch_version = _get_pytorch_version_int()

# Load metadata directly: importing lightop would require built extensions.
_core = runpy.run_path(
    str(Path(__file__).resolve().parent / "python/lightop/core.py")
)
MODULES = _core["MODULES"]
try:
    LIGHTOP_MODULES = _core["select_modules"](os.environ)
except ValueError as exc:
    raise SystemExit(str(exc)) from None
BUILD_FULL = not LIGHTOP_MODULES
BUILD_MODULES = LIGHTOP_MODULES or tuple(MODULES)

REPO_ROOT = Path(__file__).resolve().parent


def _load_build_fingerprint():
    """Load build_fingerprint.py by path so it works from any CWD/sys.path."""
    path = REPO_ROOT / "build_fingerprint.py"
    spec = importlib.util.spec_from_file_location("lightop_build_fingerprint", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Incremental-build helpers (content-hash module fingerprints, layer 1 of §8).
_build_fp = _load_build_fingerprint()


def _find_rocm_home() -> Optional[str]:
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        try:
            pipe_hipcc = subprocess.Popen(
                ["which hipcc | xargs readlink -f"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            hipcc, _ = pipe_hipcc.communicate()
            rocm_home = os.path.dirname(os.path.dirname(hipcc.decode().rstrip('\r\n')))
            if os.path.basename(rocm_home) == 'hip':
                rocm_home = os.path.dirname(rocm_home)
        except Exception:
            rocm_home = '/opt/rocm'
            if not os.path.exists(rocm_home):
                rocm_home = None
    if rocm_home and torch.version.hip is None:
        print(f"No ROCm runtime is found, using ROCM_HOME='{rocm_home}'")
    return rocm_home


def _get_rocm_arch_flags(cflags: Optional[List[str]] = None) -> List[str]:
    if cflags is not None:
        for flag in cflags:
            if 'amdgpu-target' in flag:
                return ['-fno-gpu-rdc']
    dtk_ver = int(str(dtk_version)[:4])  # dtk命名问题25042和2604  取前四位比较
    if dtk_ver >= 2604:  # dtk>=26042 can support gfx938
        #archs = os.environ.get('PYTORCH_ROCM_ARCH', 'gfx906;gfx926;gfx928;gfx936;gfx938')
        default_archs = 'gfx928;gfx936;gfx938'
        if dtk_ver >= 2610:
            default_archs += ';gfx92a'
        archs = os.environ.get('PYTORCH_ROCM_ARCH', default_archs)
    else:
        archs = os.environ.get('PYTORCH_ROCM_ARCH', 'gfx906;gfx926;gfx928;gfx936')
    flags = ['--offload-arch=%s' % arch for arch in archs.split(';')]
    flags += ['-fno-gpu-rdc']
    flags +=['--gpu-max-threads-per-block=1024']
    return flags

def _get_rocm_arch_flags_marlin_w4a8(cflags: Optional[List[str]] = None) -> List[str]:
    dtk_ver = int(str(dtk_version)[:4])
    if dtk_ver >= 2604:  # dtk>=26042 can support gfx938
        default_archs = 'gfx906;gfx926;gfx928;gfx936;gfx938'
        if dtk_ver >= 2610:
            default_archs += ';gfx92a'
        archs = os.environ.get('PYTORCH_ROCM_ARCH', default_archs)
    else:
        archs = os.environ.get('PYTORCH_ROCM_ARCH', 'gfx906;gfx926;gfx928;gfx936')
    flags = ['--offload-arch=%s' % arch for arch in archs.split(';')]
    flags +=['--gpu-max-threads-per-block=512']
    flags +=['-mllvm']
    flags +=['-enable-num-vgprs-512=true']
    # flags +=['--save-temps']
    return flags
# 4 warp
def _get_rocm_arch_flags_marlin_w4a16(cflags: Optional[List[str]] = None) -> List[str]:
    dtk_ver = int(str(dtk_version)[:4])
    if dtk_ver >= 2604:  # dtk>=26042 can support gfx938
        default_archs = 'gfx906;gfx926;gfx928;gfx936;gfx938'
        if dtk_ver >= 2610:
            default_archs += ';gfx92a'
        archs = os.environ.get('PYTORCH_ROCM_ARCH', default_archs)
    else:
        archs = os.environ.get('PYTORCH_ROCM_ARCH', 'gfx906;gfx926;gfx928;gfx936')
    flags = ['--offload-arch=%s' % arch for arch in archs.split(';')]
    flags +=['--gpu-max-threads-per-block=256']
    # flags +=['-mllvm']
    # flags +=['-enable-num-vgprs-512=true']
    # flags +=['--save-temps']
    return flags


ROCM_HOME = _find_rocm_home()
IS_HIP_EXTENSION = True if ((ROCM_HOME is not None) and (torch.version.hip is not None)) else False
COMMON_HIP_FLAGS = [
    '-fPIC',
    '-D__HIP_PLATFORM_HCC__=1',
    '-D__HIP_PLATFORM_AMD__=1',
    '-DUSE_ROCM=1',
]

COMMON_HIPCC_FLAGS = [
    '-DCUDA_HAS_FP16=1',
]


def is_ninja_available():
    try:
        subprocess.check_output('ninja --version'.split())
    except Exception:
        return False
    else:
        return True


def verify_ninja_availability():
    if not is_ninja_available():
        raise RuntimeError("Ninja is required to load C++ extensions")


def _is_cuda_file(path: str) -> bool:
    valid_ext = ['.cu', '.cuh']
    if IS_HIP_EXTENSION:
        valid_ext.append('.hip')
    return os.path.splitext(path)[1] in valid_ext


def _join_rocm_home(*paths) -> str:
    if ROCM_HOME is None:
        raise EnvironmentError('ROCM_HOME environment variable is not set. ')
    return os.path.join(ROCM_HOME, *paths)

def get_rocm_version():
    version_str = torch.version.hip
    if version_str is None:
        return None
    parts = [int(x) for x in version_str.split('.')]
    version = parts[0] * 10000 + parts[1] * 100 + int((str(parts[2])[:2]) if len(parts) > 2 else 0)
    # version = parts[0] * 10000 + parts[1] * 100
    return version

ROCM_VERSION = get_rocm_version()

def _write_ninja_file(path, cflags, post_cflags, cuda_cflags, cuda_post_cflags, sources,
                      objects, ldflags, library_target, with_cuda) -> None:
    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    ldflags = sanitize_flags(ldflags)
    assert len(sources) == len(objects)
    assert len(sources) > 0
    compiler = os.environ.get('CXX', 'c++')
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')
    if with_cuda:
        if IS_HIP_EXTENSION:
            nvcc = _join_rocm_home('bin', 'hipcc')
        config.append(f'nvcc = {nvcc}')
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')
    sources = [os.path.abspath(file) for file in sources]
    compile_rule = ['rule compile']
    compile_rule.append('  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
    compile_rule.append('  description = Compiling $in')
    compile_rule.append('  depfile = $out.d')
    compile_rule.append('  deps = gcc')
    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        required_cuda_version = version.parse('10.2')
        has_cuda_version = torch.version.cuda is not None
        # HIP/DTK leaves torch.version.cuda as None, so the stock guard would
        # disable header dependency tracking entirely and a .cuh/.h edit would
        # never invalidate a stale .o (analysis §2.3). hipcc is clang based and
        # supports -MMD -MF, so enable depfile tracking for HIP as well.
        headerdeps_supported = IS_HIP_EXTENSION or (
            has_cuda_version and version.parse(torch.version.cuda) >= required_cuda_version)
        if headerdeps_supported:
            nvcc_gendeps = '-MMD -MF $out.d'
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
        cuda_compile_rule.append(
            f'  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')
        cuda_compile_rule.append('  description = Compiling $in')
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        rule = 'cuda_compile' if is_cuda_source else 'compile'
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')
    if library_target is not None:
        link_rule = ['rule link']
        link_rule.append('  command = $cxx $in $ldflags -o $out')
        link_rule.append('  description = Linking $out')
        link = [f'build {library_target}: link {" ".join(objects)}']
        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)
    blocks += [link_rule, build, link, default]
    with open(path, 'w') as build_file:
        for block in blocks:
            lines = '\n'.join(block)
            build_file.write(f'{lines}\n\n')


def _get_num_workers(verbose: bool) -> Optional[int]:
    max_jobs = os.environ.get('MAX_JOBS') #设置编译的线程数
    #max_jobs = '32'
    if max_jobs is not None and max_jobs.isdigit():
        if verbose:
            print(f'Using envvar MAX_JOBS ({max_jobs}) as the number of workers...')
        return int(max_jobs)
    if verbose:
        print('Allowing ninja to set a default number of workers... ')
    return None


def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    # Keep compiler flags, including the real HIP platform macro, in the build
    # itself while preventing the full command line from reaching user output.
    command = ['ninja']
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(['-j', str(num_workers)])
    env = os.environ.copy()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                cwd=build_directory, check=False, env=env)
        output = result.stdout or b''
        output = output.replace(b'-D__HIP_PLATFORM_AMD__=1', b'-D<platform-macro-redacted>=1')
        output = output.replace(b'__HIP_PLATFORM_AMD__', b'<platform-macro-redacted>')
        if verbose and output:
            sys.stdout.write(output.decode('utf-8', 'replace'))
            sys.stdout.flush()
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command, output=output)
    except subprocess.CalledProcessError as e:
        _, error, _ = sys.exc_info()
        message = error_prefix
        if hasattr(error, 'output') and error.output:  # type: ignore[union-attr]
            message += f": {error.output.decode('utf-8', 'replace')}"  # type: ignore[union-attr]
        raise RuntimeError(message) from e

def _write_ninja_file_and_compile_objects(sources: List[str], objects, cflags, post_cflags, cuda_cflags,
                                          cuda_post_cflags, build_directory: str, verbose: bool,
                                          with_cuda: Optional[bool]) -> None:
    verify_ninja_availability()
    compiler = os.environ.get('CXX', 'c++')
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    build_file_path = os.path.join(build_directory, 'build.ninja')
    if verbose:
        print(f'Emitting ninja build file {build_file_path}...')
    _write_ninja_file(path=build_file_path, cflags=cflags, post_cflags=post_cflags, cuda_cflags=cuda_cflags,
                      cuda_post_cflags=cuda_post_cflags, sources=sources, objects=objects, ldflags=None,
                      library_target=None, with_cuda=with_cuda)
    if verbose:
        print('Compiling objects...')
    _run_ninja_build(
        build_directory,
        verbose,
        error_prefix='Error compiling objects for extension')

min_supported_cpython = "0x03090000"  # Python 3.9 hexcode

class BuildReleaseExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        super(BuildReleaseExtension, self).__init__(*args, **kwargs)

    def _add_compile_flag(self, extension, flag):
        # setuptools can rerun build_ext when building in-place and a wheel in
        # one invocation. Keep PyTorch's injected flags stable across runs.
        args = extension.extra_compile_args
        groups = args.values() if isinstance(args, dict) else (args,)
        for flags in groups:
            if flag not in flags:
                flags.append(flag)

    def run(self):
        super().run()
        # Only a full build owns the whole build_lib tree. A selective build
        # (LIGHTOP_MODULES set) builds a subset of the extensions, so its
        # "expected" set must never be used to delete the other families'
        # artifacts - that would wipe _C/mqa_logits/... from build_lib every time
        # a single module is rebuilt in-place. Skip pruning entirely when
        # selective; a full build already sees every extension as expected.
        if not self.dry_run and not LIGHTOP_MODULES:
            # Only this build's extensions belong in a wheel. In-place source
            # extensions are kept so rebuilding one family preserves the others.
            expected = {Path(self.build_lib, self.get_ext_filename(ext.name)).resolve()
                        for ext in self.extensions}
            package = Path(self.build_lib) / "lightop"
            for directory in (package, package / "_ext"):
                for artifact in directory.glob("*.so"):
                    if artifact.resolve() not in expected:
                        artifact.unlink()
        if not self.dry_run and self.extensions:
            modules = ", ".join(LIGHTOP_MODULES) if LIGHTOP_MODULES else "full"
            print(f"\n[LightOp] Build succeeded: {modules}")

    def build_extensions(self) -> None:
        self._check_abi()
        cuda_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not cuda_ext and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == '.cu':
                    cuda_ext = True
                    break
            extension = next(extension_iter, None)

        if torch_version < 290:
            for extension in self.extensions:
                if isinstance(extension.extra_compile_args, dict):
                    for ext in ['cxx', 'nvcc']:
                        if ext not in extension.extra_compile_args:
                            extension.extra_compile_args[ext] = []
                self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')
                for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
                    val = getattr(torch._C, f"_PYBIND11_{name}")
                    self._add_compile_flag(extension, f'-DPYBIND11_{name}="{val}"')
                self._define_torch_extension_name(extension)
                self._add_gnu_cpp_abi_flag(extension)
        else:
            # 从这里开始，使用与PyTorch 2.9.1完全一致的逻辑
            #IS_HIP_EXTENSION = True  # 根据你的实际情况设置
            IS_WINDOWS = False  # 根据你的实际情况设置
            
            for extension in self.extensions:
                # 注意：这里增加了'sycl'支持，与2.9.1一致
                if isinstance(extension.extra_compile_args, dict):
                    for ext in ['cxx', 'nvcc', 'sycl']:
                        if ext not in extension.extra_compile_args:
                            extension.extra_compile_args[ext] = []

                self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')
                self._add_compile_flag(extension, f'-DROCM_VERSION={ROCM_VERSION}')

                # HIP扩展处理（根据你的需要保留或删除）
                if True: #IS_HIP_EXTENSION:
                    self._hipify_compile_flags(extension)

                # Python有限API支持（根据你的需要保留或删除）
                if extension.py_limited_api:
                    # 你需要定义min_supported_cpython变量
                    # 例如：min_supported_cpython = '0x03070000'  # Python 3.7
                    self._add_compile_flag(extension, f'-DPy_LIMITED_API={min_supported_cpython}')
                
                self._define_torch_extension_name(extension)
                # 注意：这里移除了 _add_gnu_cpp_abi_flag 调用
        
        self.compiler.src_extensions += ['.cu', '.cuh', '.hip']

        def append_cxx_std_if_no_std_present(cflags) -> None:
            cpp_format_prefix = '/{}:' if self.compiler.compiler_type == 'msvc' else '-{}='
            cpp_flag_prefix = cpp_format_prefix.format('std')
            cpp_standard = (
                'c++20'
                if version.parse(torch.__version__.split('+')[0]) >= version.parse('2.14')
                else 'c++17'
            )
            cpp_flag = cpp_flag_prefix + cpp_standard
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def convert_to_absolute_paths_inplace(paths):
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_ninja_compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0,
                                    extra_preargs=None, extra_postargs=None, depends=None):
            output_dir = os.path.abspath(output_dir)
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)
            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            if (debug):
                print("debug mode")
            else:
                if('-g' in extra_cc_cflags):
                    extra_cc_cflags.remove('-g')
                if('-Wall' in extra_cc_cflags):
                    extra_cc_cflags.remove('-Wall')
                print("release mode")
            
            marlin_w4a16_dir = os.path.abspath('./csrc/moe_marlin_gemm/w4a16/')
            marlin_w4a8_dir = os.path.abspath('./csrc/moe_marlin_gemm/w4a8/')
            marlin_w4fp8_dir = os.path.abspath('./csrc/moe_marlin_gemm/w4fp8/')  # 如果也需要特殊处理

            marlin_w4a16_sources = []
            marlin_w4a8_sources = []
            marlin_w4fp8_sources = []
            other_sources = []

            for s in sources:
                abs_path = os.path.abspath(s)
                if abs_path.startswith(marlin_w4a16_dir):
                    marlin_w4a16_sources.append(s)
                elif abs_path.startswith(marlin_w4a8_dir):
                    marlin_w4a8_sources.append(s)
                elif abs_path.startswith(marlin_w4fp8_dir):
                    marlin_w4fp8_sources.append(s)
                else:
                    other_sources.append(s)
            
            marlin_w4a16_objects = [obj for src, obj in zip(sources, objects) if src in marlin_w4a16_sources]
            marlin_w4a8_objects = [obj for src, obj in zip(sources, objects) if src in marlin_w4a8_sources]
            marlin_w4fp8_objects = [obj for src, obj in zip(sources, objects) if src in marlin_w4fp8_sources]
            other_objects = [obj for src, obj in zip(sources, objects) if src in other_sources]
            
            with_cuda = any(map(_is_cuda_file, sources))
            
            # compile other
            if other_sources:
                if isinstance(extra_postargs, dict):
                    post_cflags = extra_postargs['cxx']
                else:
                    post_cflags = list(extra_postargs)
                if IS_HIP_EXTENSION:
                    post_cflags = COMMON_HIP_FLAGS + post_cflags
                append_cxx_std_if_no_std_present(post_cflags)
                
                cuda_post_cflags = None
                if with_cuda:
                    if isinstance(extra_postargs, dict):
                        cuda_post_cflags = extra_postargs['nvcc']
                    else:
                        cuda_post_cflags = list(extra_postargs)
                    if IS_HIP_EXTENSION:
                        cuda_post_cflags = cuda_post_cflags + _get_rocm_arch_flags(cuda_post_cflags)
                        cuda_post_cflags = COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS + cuda_post_cflags
                    append_cxx_std_if_no_std_present(cuda_post_cflags)
                    cuda_cflags = [shlex.quote(f) for f in common_cflags]
                    cuda_post_cflags = [shlex.quote(f) for f in cuda_post_cflags]
                
                _write_ninja_file_and_compile_objects(
                    sources=other_sources, 
                    objects=other_objects,
                    cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                    post_cflags=[shlex.quote(f) for f in post_cflags],
                    cuda_cflags=cuda_cflags if with_cuda else None,
                    cuda_post_cflags=cuda_post_cflags if with_cuda else None,
                    build_directory=output_dir,
                    verbose=True,
                    with_cuda=with_cuda
                )
            
            # compile w4a16
            if marlin_w4a16_sources:
                marlin_cxx_flags = ['-O3', '-w']
                marlin_nvcc_flags = ['-O3', '-w']
                if IS_HIP_EXTENSION:
                    marlin_cxx_flags = COMMON_HIP_FLAGS + marlin_cxx_flags
                append_cxx_std_if_no_std_present(marlin_cxx_flags)

                marlin_cuda_cflags = None
                marlin_cuda_post_cflags = None
                if with_cuda:
                    marlin_cuda_cflags = [shlex.quote(f) for f in common_cflags]
                    marlin_cuda_post_cflags = list(marlin_nvcc_flags)
                    if IS_HIP_EXTENSION:
                        marlin_cuda_post_cflags += _get_rocm_arch_flags_marlin_w4a16(marlin_cuda_post_cflags)
                        marlin_cuda_post_cflags = COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS + marlin_cuda_post_cflags
                    append_cxx_std_if_no_std_present(marlin_cuda_post_cflags)
                    marlin_cuda_post_cflags = [shlex.quote(f) for f in marlin_cuda_post_cflags]

                _write_ninja_file_and_compile_objects(
                    sources=marlin_w4a16_sources,
                    objects=marlin_w4a16_objects,
                    cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                    post_cflags=[shlex.quote(f) for f in marlin_cxx_flags],
                    cuda_cflags=marlin_cuda_cflags,
                    cuda_post_cflags=marlin_cuda_post_cflags,
                    build_directory=output_dir,
                    verbose=True,
                    with_cuda=with_cuda
                )
            # compile w4a8
            if marlin_w4a8_sources:
                marlin_cxx_flags = ['-O3', '-w']
                marlin_nvcc_flags = ['-O3', '-w']
                if IS_HIP_EXTENSION:
                    marlin_cxx_flags = COMMON_HIP_FLAGS + marlin_cxx_flags
                append_cxx_std_if_no_std_present(marlin_cxx_flags)

                marlin_cuda_cflags = None
                marlin_cuda_post_cflags = None
                if with_cuda:
                    marlin_cuda_cflags = [shlex.quote(f) for f in common_cflags]
                    marlin_cuda_post_cflags = list(marlin_nvcc_flags)
                    if IS_HIP_EXTENSION:
                        marlin_cuda_post_cflags += _get_rocm_arch_flags_marlin_w4a8(marlin_cuda_post_cflags)
                        marlin_cuda_post_cflags = COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS + marlin_cuda_post_cflags
                    append_cxx_std_if_no_std_present(marlin_cuda_post_cflags)
                    marlin_cuda_post_cflags = [shlex.quote(f) for f in marlin_cuda_post_cflags]

                _write_ninja_file_and_compile_objects(
                    sources=marlin_w4a8_sources,
                    objects=marlin_w4a8_objects,
                    cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                    post_cflags=[shlex.quote(f) for f in marlin_cxx_flags],
                    cuda_cflags=marlin_cuda_cflags,
                    cuda_post_cflags=marlin_cuda_post_cflags,
                    build_directory=output_dir,
                    verbose=True,
                    with_cuda=with_cuda
                )
            # compile w4fp8
            if marlin_w4fp8_sources:
                marlin_cxx_flags = ['-O3', '-w']
                marlin_nvcc_flags = ['-O3', '-w']
                if IS_HIP_EXTENSION:
                    marlin_cxx_flags = COMMON_HIP_FLAGS + marlin_cxx_flags
                append_cxx_std_if_no_std_present(marlin_cxx_flags)

                marlin_cuda_cflags = None
                marlin_cuda_post_cflags = None
                if with_cuda:
                    marlin_cuda_cflags = [shlex.quote(f) for f in common_cflags]
                    marlin_cuda_post_cflags = list(marlin_nvcc_flags)
                    if IS_HIP_EXTENSION:
                        marlin_cuda_post_cflags += _get_rocm_arch_flags_marlin_w4a8(marlin_cuda_post_cflags)
                        marlin_cuda_post_cflags = COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS + marlin_cuda_post_cflags
                    append_cxx_std_if_no_std_present(marlin_cuda_post_cflags)
                    marlin_cuda_post_cflags = [shlex.quote(f) for f in marlin_cuda_post_cflags]

                _write_ninja_file_and_compile_objects(
                    sources=marlin_w4fp8_sources,
                    objects=marlin_w4fp8_objects,
                    cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                    post_cflags=[shlex.quote(f) for f in marlin_cxx_flags],
                    cuda_cflags=marlin_cuda_cflags,
                    cuda_post_cflags=marlin_cuda_post_cflags,
                    build_directory=output_dir,
                    verbose=True,
                    with_cuda=with_cuda
                )
            
            return other_objects + marlin_w4a16_objects + marlin_w4a8_objects + marlin_w4fp8_objects

        self.compiler.compile = unix_wrap_ninja_compile
        build_ext.build_extensions(self)

    def finalize_options(self):
        # torch's BuildExtension.finalize_options() forces self.force = True when
        # Ninja is enabled, which would defeat the per-module fingerprint skip.
        # Capture the user's real force intent *before* delegating (analysis §8.4).
        self._lightop_user_force = bool(getattr(self, "force", False))
        super().finalize_options()

    # ----- layer 1: content-hash fingerprint skip -------------------------
    def _lightop_build_signature(self, ext):
        """Everything except file contents that must invalidate a module build."""
        extra = ext.extra_compile_args
        if isinstance(extra, dict):
            cxx_flags = list(extra.get("cxx", []))
            nvcc_flags = list(extra.get("nvcc", []))
        else:
            cxx_flags = list(extra)
            nvcc_flags = []
        return {
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "torch_hip": str(torch.version.hip),
            "dtk": str(dtk_version),
            "rocm_home": str(ROCM_HOME),
            "rocm_version": str(ROCM_VERSION),
            "is_hip": bool(IS_HIP_EXTENSION),
            "arch_env": os.environ.get("PYTORCH_ROCM_ARCH", ""),
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
            "include_dirs": [os.path.abspath(path) for path in (ext.include_dirs or [])],
            "libraries": list(ext.libraries or []),
            "extra_objects": list(getattr(ext, "extra_objects", []) or []),
            "define_macros": sorted((str(macro[0]), str(macro[1]) if len(macro) > 1 else None)
                                    for macro in (ext.define_macros or [])),
        }

    def _lightop_watch_files(self, ext, module_build_dir, declared_headers):
        """Repository-local files whose contents determine a module rebuild.

        Sources always count. Headers come from (a) the exact dependency set the
        previous hipcc ``-MMD`` run recorded - which Ninja keeps in its binary
        ``.ninja_deps`` database and *not* in ``*.o.d`` files - (b) any
        registry-declared header globs (a safety net before the first build),
        and (c) every shared header under ``csrc/include`` because hipify may
        rename those (``cuda_meta.h`` -> ``hip_meta.h``, ``reduce.h`` ->
        ``reduce_hip.h``) so the recorded dependency alone can miss the original
        file a developer actually edits.
        """
        repo_root = str(REPO_ROOT)
        files = set()

        def _absolute(path):
            return path if os.path.isabs(path) else os.path.join(repo_root, path)

        for source in ext.sources or []:
            files.add(os.path.realpath(_absolute(source)))
        for pattern in declared_headers or []:
            for match in glob.glob(_absolute(pattern), recursive=True):
                files.add(os.path.realpath(match))
        # ``*.o.d`` files are usually consumed and deleted by Ninja, so this only
        # helps on toolchains that leave them behind; the ``.ninja_deps`` dump is
        # the authoritative dependency source.
        files.update(_build_fp.collect_dependency_files([module_build_dir], repo_root))
        files.update(_build_fp.collect_ninja_deps(module_build_dir, repo_root))
        # Shared headers under csrc/include are only relevant to the modules
        # whose objects actually depend on them. Gate them by the dependency
        # names Ninja recorded (in .ninja_deps) so that editing one shared header
        # rebuilds exactly the modules that include it - hipify may rename the
        # recorded file, so the check tolerates the _hip / cuda_->hip_ spellings.
        dependency_names = _build_fp.ninja_dependency_names(module_build_dir)
        include_root = os.path.join(repo_root, "csrc", "include")
        for suffix in ("*.h", "*.hpp", "*.cuh"):
            for match in glob.glob(os.path.join(include_root, suffix)):
                if not os.path.isfile(match):
                    continue
                if _build_fp.shared_header_referenced(os.path.basename(match), dependency_names):
                    files.add(os.path.realpath(match))
        return files

    def _lightop_is_built(self, ext):
        """True if the extension's ``.so`` exists anywhere a build could place it.

        ``get_ext_fullpath`` resolves to the source tree under ``--inplace`` and
        to ``build_lib`` otherwise; checking every plausible location keeps the
        skip decision correct for both flows and for installs with a custom
        ``package_dir`` (``python/lightop``).
        """
        candidates = []
        try:
            candidates.append(self.get_ext_fullpath(ext.name))
        except Exception:
            pass
        try:
            filename = self.get_ext_filename(ext.name)
            candidates.append(os.path.join(self.build_lib, filename))
            package_dir = self.get_finalized_command("build_py").get_package_dir("lightop")
            if package_dir:
                parts = ext.name.split(".")
                candidates.append(os.path.join(package_dir, *parts[1:-1], filename))
        except Exception:
            pass
        return any(candidate and os.path.exists(candidate) for candidate in candidates)

    def _lightop_should_skip(self, ext, key, build_root, module_build_dir, declared_headers):
        label = getattr(ext, "_lightop_name", key)
        if getattr(self, "_lightop_user_force", False) or \
                os.environ.get("LIGHTOP_FORCE_REBUILD", "0") == "1":
            print(f"[LightOp] module '{label}': forced rebuild")
            return False
        if not self._lightop_is_built(ext):
            return False
        cache_path = _build_fp.fingerprint_path(build_root, key)
        previous = _build_fp.load_fingerprint(cache_path)
        if previous is None:
            return False
        signature = self._lightop_build_signature(ext)
        watch = self._lightop_watch_files(ext, module_build_dir, declared_headers)
        fingerprint, entries = _build_fp.compute_fingerprint(str(REPO_ROOT), watch, signature)
        if fingerprint == previous.get("fingerprint") and previous.get("build") == signature:
            return True
        # The hash caught a real content change; make sure Ninja's mtime graph
        # also sees it, because a ``git checkout`` can leave a changed file with
        # an older mtime than the cached object (analysis §8.1). Touching only
        # the changed files keeps layer 2 as fine-grained as possible.
        previous_files = previous.get("files") or {}
        for relpath, digest in entries.items():
            if previous_files.get(relpath) != digest:
                try:
                    os.utime(os.path.join(str(REPO_ROOT), relpath), None)
                except OSError:
                    pass
        reason = _build_fp.describe_changes(previous, entries, signature)
        print(f"[LightOp] module '{label}': rebuilding ({reason})")
        return False

    def _lightop_store_fingerprint(self, ext, key, build_root, module_build_dir, declared_headers):
        signature = self._lightop_build_signature(ext)
        watch = self._lightop_watch_files(ext, module_build_dir, declared_headers)
        fingerprint, entries = _build_fp.compute_fingerprint(str(REPO_ROOT), watch, signature)
        _build_fp.write_fingerprint(_build_fp.fingerprint_path(build_root, key), {
            "key": key,
            "module": getattr(ext, "_lightop_name", key),
            "fingerprint": fingerprint,
            "build": signature,
            "files": entries,
        })

    def build_extension(self, ext):
        # Keep each extension's object cache independent of the build selection:
        # every module gets its own build_temp/<ext> ninja directory (layer 2).
        key = getattr(ext, "_lightop_key", None)
        declared_headers = getattr(ext, "_lightop_headers", []) or []
        orig_temp = self.build_temp
        build_root = os.path.abspath(orig_temp)
        module_build_dir = os.path.join(build_root, str(ext.name).replace('.', '__'))
        if not self.dry_run and key is not None:
            try:
                if self._lightop_should_skip(ext, key, build_root,
                                             module_build_dir, declared_headers):
                    print(f"[LightOp] module '{getattr(ext, '_lightop_name', key)}': "
                          f"up to date, skipping ({ext.name})")
                    return
            except Exception as exc:  # bookkeeping must never break the build
                print(f"[LightOp] fingerprint check for {ext.name} failed: {exc}")
        self.build_temp = module_build_dir
        try:
            super().build_extension(ext)
        finally:
            self.build_temp = orig_temp
        if not self.dry_run and key is not None:
            try:
                self._lightop_store_fingerprint(ext, key, build_root,
                                                module_build_dir, declared_headers)
            except Exception as exc:
                print(f"[LightOp] failed to record fingerprint for {ext.name}: {exc}")


def torch_version_over110():
    version=torch.__version__.split('.')
    return int(version[0])*100+int(version[1])>110

def torch_version_over2120():
    version=torch.__version__.split('.')
    return int(version[0])*100+int(version[1])>=212

def _mark_incremental(ext, key, name, headers=None):
    """Attach the metadata BuildReleaseExtension needs for the fingerprint skip."""
    ext._lightop_key = key
    ext._lightop_name = name
    ext._lightop_headers = list(headers or [])
    return ext


def get_extensions():
    extensions = []
    include_dirs = []
    define_macros = []

    # 全局默认编译选项
    extra_compile_args = {'cxx': ['-O3','-ldl', '-lrocblas','-w'], 'nvcc': ['-O3','-ldl', '-lrocblas','-w','-mllvm -enable-num-vgprs-512=true', '-DHIP_ENABLE_WARP_SYNC_BUILTINS']}
    if torch_version_over110():
        extra_compile_args['nvcc'].append('-DTORCH_VERSION_OVER_110')
    if torch_version >= 290:
        extra_compile_args['nvcc'].append('-DTORCH_VERSION_OVER_290')
    if torch_version_over2120():
        extra_compile_args['nvcc'].append('-DTORCH_VERSION_OVER_2120')
    # 源码清单：按算子家族显式列出，不含递归 glob。
    # 不用 csrc/**/*.cu 是因为 moe_marlin_gemm/w4a8/awq_marlin_repack_w4a8.cu
    # 是被 #include 引入的实现文件，递归 glob 会把它当成独立编译单元。
    _CSRC_DIRS = [
        ('asm', '*.cu'),
        ('awq_w4a16_gemm', '*.cu'),
        ('common', '*.cpp'),
        ('kernels', '*.cu'),
        ('kernels/blockwise_scaling', '*.cu'),
        ('kernels/bmm_scale', '*.cu'),
        ('kernels/concat', '*.cu'),
        ('kernels/cp_output', '*.cu'),
        ('kernels/gated_activation', '*.cu'),
        ('kernels/grid_sample', '*.cu'),
        ('kernels/groupnorm', '*.cu'),
        ('kernels/index_k_int8_dequant', '*.cu'),
        ('kernels/kvcache', '*.cu'),
        ('kernels/layernorm', '*.cu'),
        ('kernels/loss', '*.cu'),
        ('kernels/mla', '*.cu'),
        ('kernels/moe_gate', '*.cu'),
        ('kernels/per_token_quant', '*.cu'),
        ('kernels/polyfill', '*.cu'),
        ('kernels/rmsnorm', '*.cu'),
        ('kernels/sampling', '*.cu'),
        ('kernels/scaledmask', '*.cu'),
        ('kernels/softmax', '*.cu'),
        ('kernels/topk', '*.cu'),
        ('lmslimquant_native', '*.cpp'),
        ('lmslimquant_native/awq', '*.cu'),
        ('lmslimquant_native/gptq', '*.cu'),
        ('lmslimquant_native/hipblaslt', '*.cu'),
        ('lmslimquant_native/smoothquant', '*.cpp'),
        ('lmslimquant_native/torch_bridge', '*.cu'),
        ('moe_gemm/w16a16', '*.cu'),
        ('moe_gemm/w4a8', '*.cu'),
        ('moe_gemm/w8a8', '*.cu'),
        ('moe_marlin_gemm/w4a16', '*.cu'),
        ('moe_marlin_gemm/w4a8', '*.cu'),
        ('moe_marlin_gemm/w4fp8', '*.cu'),
        ('mqa_logits', '*.cu'),
        ('mqa_logits/bf16', '*.cu'),
        ('mqa_logits/fp8', '*.cu'),
        ('mqa_logits/fp8_fp4', '*.cu'),
        ('mqa_logits/int8', '*.cu'),
        ('mqa_logits/paged', '*.cu'),
        ('mqa_logits/sparse', '*.cu'),
        ('pybind', '*.cpp'),
    ]

    module_sources = {}
    for name in BUILD_MODULES:
        sources = []
        for pattern in MODULES[name]["sources"]:
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise ValueError(f"{name}: source pattern matched no files: {pattern}")
            sources.extend(matches)
        module_sources[name] = list(dict.fromkeys(sources))

    # Headers the monolith (_C) may include. ``unix_wrap_ninja_compile`` compiles
    # several source groups into one build directory and only the last group's
    # dependency records survive in ``.ninja_deps``, so dependency-based header
    # watching is unreliable for _C. Instead watch every header located in the
    # monolith's own source directories, skipping directories owned by migrated
    # modules (their headers are already watched per module).
    migrated_dirs = {os.path.dirname(os.path.abspath(source))
                     for sources in module_sources.values() for source in sources}
    monolith_header_globs = []
    for _header_dir in [f'./csrc/{_dir}' for _dir, _ in _CSRC_DIRS] + ['./csrc/include']:
        if os.path.abspath(_header_dir) in migrated_dirs:
            continue
        for _pattern in ('*.h', '*.hpp', '*.cuh'):
            monolith_header_globs.append(os.path.join(_header_dir, _pattern))

    indexer_prep = "./csrc/kernels/mla/dsv41_indexer_prep.cu"
    op_files = []
    if BUILD_FULL:
        for _src_dir, _src_pat in _CSRC_DIRS:
            op_files += glob.glob(f'./csrc/{_src_dir}/{_src_pat}')
        # This source defines a standalone extension with stricter FP contraction.
        migrated = {str(Path(source).resolve())
                    for sources in module_sources.values() for source in sources}
        migrated.add(str(Path(indexer_prep).resolve()))
        op_files = [source for source in op_files
                    if str(Path(source).resolve()) not in migrated]

    extension = CUDAExtension
    # 迁移后只需要两个搜索根：csrc/ 与 csrc/include/。
    # 跨目录引用一律写成显式相对路径，不再为每个子目录单列一条。
    include_dirs.append(os.path.abspath('./csrc/'))
    include_dirs.append(os.path.abspath('./csrc/include/'))
    rocm_home=_find_rocm_home()
    include_dirs.append(rocm_home+'/hiprand/include')
    include_dirs.append(rocm_home+'/rocrand/include')
    
    if BUILD_FULL:
        # Remaining operators; migrated families are built separately below.
        ext_ops = extension(
            name="lightop._C",
            sources=op_files,
            include_dirs=list(include_dirs),
            libraries=['rocblas', 'hipblas', 'hipblaslt'],
            library_dirs=[os.path.join(rocm_home, 'lib')],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args)

        extensions.append(_mark_incremental(ext_ops, "_C", "_C", monolith_header_globs))
        extensions.append(_mark_incremental(extension(
            name="lightop._router_gemm",
            sources=["./csrc/router/router_gemm.cu"],
            extra_compile_args={"nvcc": ["-O3"]},
        ), "_router_gemm", "_router_gemm"))
        extensions.append(_mark_incremental(extension(
            name="lightop._dsv41_prep",
            sources=[indexer_prep],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-ffp-contract=off"]},
        ), "_dsv41_prep", "_dsv41_prep"))

    # Each family supplies only its additions to the common build settings.
    for name in BUILD_MODULES:
        spec = MODULES[name]
        sources = module_sources[name]
        module_compile_args = {k: list(v) for k, v in extra_compile_args.items()}
        for compiler, flags in spec["extra_compile_args"].items():
            module_compile_args.setdefault(compiler, []).extend(flags)
        extensions.append(_mark_incremental(extension(
            name=spec["extension"],
            sources=sources,
            include_dirs=list(include_dirs) + [
                os.path.abspath(path) for path in spec["include_dirs"]
            ],
            libraries=['rocblas', 'hipblas', 'hipblaslt'],
            library_dirs=[os.path.join(rocm_home, 'lib')],
            define_macros=define_macros,
            extra_compile_args=module_compile_args,
        ), "module:" + name, name, spec.get("headers")))

    return extensions

def _get_pytorch_version():
    if "PYTORCH_VERSION" in os.environ:
        return f"{os.environ['PYTORCH_VERSION']}"
    return torch.__version__


pytorch_dep = 'torch'
if os.getenv('PYTORCH_VERSION'):
    pytorch_dep += "==" + os.getenv('PYTORCH_VERSION')

PACKAGE_RESOURCE_DIRS = {
    "configs": "configs",
    "hsa": "hsa",
    "lib": "lib",
    "third_party/librarygemmbiasf": "librarygemmbiasf",
    "third_party/librarygemmdbiasb": "librarygemmdbiasb",
}


class BuildPyWithResources(build_py):
    """Copy repository-level runtime resources into the installed package."""

    def run(self):
        super().run()
        package_root = os.path.join(self.build_lib, "lightop")
        for source, destination in PACKAGE_RESOURCE_DIRS.items():
            # Static archives are build inputs only. Keep them in the repository's
            # shared lib/ tree without copying them into the installed wheel.
            ignore = shutil.ignore_patterns("static") if source == "lib" else None
            shutil.copytree(
                source,
                os.path.join(package_root, destination),
                dirs_exist_ok=True,
                ignore=ignore,
            )
        # Legacy egg installs do not copy license_files into EGG-INFO. Keep
        # these materials in the package as well as wheel license metadata.
        license_root = os.path.join(package_root, "licenses")
        os.makedirs(license_root, exist_ok=True)
        for source in self.distribution.metadata.license_files or []:
            self.copy_file(source, os.path.join(license_root, os.path.basename(source)))


setup(
    name='lightop',
    version=get_version(ROCM_HOME),
    description='Torch Fast Operator Foundation',
    keywords='CV, NLP',
    license_files=[
        'LICENSE',
        'THIRD_PARTY_NOTICES.md',
        'third_party/licenses/LICENSE-aiter',
        'third_party/librarygemmbiasf/LICENSE-half-2.1.0',
    ],
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    include_package_data=False,
    package_data={
        'lightop': [
            "module_registry.json",
            "licenses/*",
            "lib/*",
            "librarygemmbiasf/*",
            "librarygemmdbiasb/*",
            "hsa/*",
            "hsa/*/*",
            "hsa/*/*/*",
            "hsa/*/*/*/*",
            "configs/*"
        ]
    },
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildReleaseExtension,
        'build_py': BuildPyWithResources,
    },
    zip_safe=False,
    install_requires=[
         pytorch_dep,
     ],
)
