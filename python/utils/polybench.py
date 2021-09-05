""" Utitity functions.  """

import datetime
import functools
import glob
import itertools
import json
import logging
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from multiprocessing import Pool
from timeit import default_timer as timer
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

POLYBENCH_DATASETS = ("MINI", "SMALL", "LARGE", "EXTRALARGE")
POLYBENCH_EXAMPLES = (
    "2mm",
    "3mm",
    "adi",
    "atax",
    "bicg",
    "cholesky",
    "correlation",
    "covariance",
    "deriche",
    "doitgen",
    "durbin",
    "fdtd-2d",
    "floyd-warshall",
    "gemm",
    "gemver",
    "gesummv",
    "gramschmidt",
    "heat-3d",
    "jacobi-1d",
    "jacobi-2d",
    "lu",
    "ludcmp",
    "mvt",
    "nussinov",
    "seidel-2d",
    "symm",
    "syr2k",
    "syrk",
    "trisolv",
    "trmm",
)
RESOURCE_FIELDS = ("DSP", "FF", "LUT", "BRAM_18K", "URAM")
RECORD_FIELDS = (
    "name",
    "run_status",
    "latency",
    "syn_latency",
    "res_usage",
    "res_avail",
)
RUN_STATUS_FIELDS = ("phism_synth", "tbgen_cosim", "phism_cosim")

PHISM_VITIS_STEPS = ("phism", "tbgen", "cosim")

Record = namedtuple("Record", RECORD_FIELDS)
Resource = namedtuple("Resource", RESOURCE_FIELDS)
RunStatus = namedtuple("RunStatus", RUN_STATUS_FIELDS)


@dataclass
class PbFlowOptions:
    """An interface for the CLI options."""

    pb_dir: str
    job: int
    polymer: bool
    cosim: bool
    debug: bool
    dataset: str
    cleanup: bool
    work_dir: str = ""
    dry_run: bool = False
    examples: List[str] = POLYBENCH_EXAMPLES
    split: str = "NO_SPLIT"  # other options: "SPLIT", "HEURISTIC"
    loop_transforms: bool = True
    constant_args: bool = True
    improve_pipelining: bool = True
    max_span: int = -1


# ----------------------- Utility functions ------------------------------------


def get_timestamp():
    """Get the current timestamp."""
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def get_project_root():
    """Get the root directory of the project."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def matched(s, patterns):
    """Check if string s matches any of the patterns."""
    if not patterns:
        return False
    for p in patterns:
        if p in s:
            return True
    return False


def get_single_file_with_ext(d, ext, includes=None, excludes=None):
    """Find the single file under the current directory with a specific extension."""
    for f in os.listdir(d):
        if not f.endswith(ext):
            continue
        if includes and not matched(f, includes):
            continue
        if excludes and matched(f, excludes):
            continue
        return f

    return None


def get_vitis_log(d, step, stream):
    """Return the file path to a specific Vitis log."""
    assert step in PHISM_VITIS_STEPS
    assert stream in ("stdout", "stderr")

    return os.path.join(
        d, "{step}.vitis_hls.{stream}.log".format(step=step, stream=stream)
    )


# ----------------------- Data record fetching functions -----------------------


def fetch_resource_usage(d, avail=False):
    """Find the report file *.xml and return the resource usage estimation."""
    syn_report_dir = os.path.join(d, "proj", "solution1", "syn", "report")
    if not os.path.isdir(syn_report_dir):
        return None

    syn_report = get_single_file_with_ext(syn_report_dir, "xml", ["kernel"], ["PE"])
    if not syn_report:
        return None

    syn_report = os.path.join(syn_report_dir, syn_report)
    if not os.path.isfile(syn_report):
        return None

    # Parse the XML report and find every resource usage (tags given by RESOURCE_FIELDS)
    root = ET.parse(syn_report).getroot()
    res_tag = "Resources" if not avail else "AvailableResources"
    return Resource(
        *[
            int(root.findtext("AreaEstimates/{}/{}".format(res_tag, res)))
            for res in RESOURCE_FIELDS
        ]
    )


def fetch_latency(d):
    """Fetch the simulated latency, measured in cycles."""
    tb_sim_report_dir = os.path.join(d, "tb", "solution1", "sim", "report")
    if not os.path.isdir(tb_sim_report_dir):
        return None

    tb_sim_report = get_single_file_with_ext(tb_sim_report_dir, "rpt")
    if not tb_sim_report:
        return None

    tb_sim_report = os.path.join(tb_sim_report_dir, tb_sim_report)
    if not os.path.isfile(tb_sim_report):
        return None

    latency = None
    with open(tb_sim_report, "r") as f:
        for line in f.readlines():
            comps = [x.strip() for x in line.strip().split("|")]

            # there are 9 columns, +2 before & after |
            # the 2nd column should give PASS.
            if len(comps) == 11 and comps[2].upper() == "PASS":
                latency = comps[-2]  # from the last column.

    # The report is malformed.
    if not latency:
        return None

    # Will raise error if latency is not an integer.
    return int(latency)


def fetch_syn_latency(d):
    """Fetch latency measured in the synthesis phase."""
    syn_report_dir = os.path.join(d, "proj", "solution1", "syn", "report")
    if not os.path.isdir(syn_report_dir):
        return None

    syn_report = get_single_file_with_ext(syn_report_dir, "xml", ["kernel"], ["PE"])
    if not syn_report:
        return None

    syn_report = os.path.join(syn_report_dir, syn_report)
    if not os.path.isfile(syn_report):
        return None

    # Parse the XML report and find every resource usage (tags given by RESOURCE_FIELDS)
    root = ET.parse(syn_report).getroot()
    latency = root.findtext(
        "PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency"
    )
    try:
        return int(latency)
    except:
        return None


def fetch_run_status(d):
    """Gather the resulting status of each stage."""

    def parse_synth_log(fp):
        if not os.path.isfile(fp):
            return "NO_LOG"
        else:
            with open(fp, "r") as f:
                has_error = any(
                    ("Synthesizability check failed." in line) for line in f.readlines()
                )
                if has_error:
                    return "CANNOT_SYNTH"
        return "SUCCESS"

    def parse_cosim_log(fp):
        if not os.path.isfile(fp):
            return "NO_LOG"
        else:
            with open(fp, "r") as f:
                has_error = any(
                    ("co-simulation finished: FAIL" in line) for line in f.readlines()
                )
                if has_error:
                    return "COSIM_FAILED"
        return "SUCCESS"

    return RunStatus(
        parse_synth_log(get_vitis_log(d, "phism", "stdout")),
        parse_cosim_log(get_vitis_log(d, "tbgen", "stdout")),
        parse_cosim_log(get_vitis_log(d, "cosim", "stdout")),
    )


def process_directory(d):
    """Process the result data within the given directory. Return a dictionary of all available data entries."""
    example_name = os.path.basename(d)
    return Record(
        example_name,
        fetch_run_status(d),
        fetch_latency(d),
        fetch_syn_latency(d),
        fetch_resource_usage(d),
        fetch_resource_usage(d, avail=True),
    )


def process_pb_flow_result_dir(d: str, options: PbFlowOptions):
    """Process the result directory from pb-flow runs."""
    records = []
    assert os.path.isdir(d)

    # Each example should have their original .c/.h files. We will look for that.
    pattern = "{}/**/*.h".format(d)
    for src_header_file in glob.glob(pattern, recursive=True):
        basename = os.path.basename(src_header_file)[:-2]  # skip '.h'
        if basename in options.examples:
            records.append(
                process_directory(os.path.abspath(os.path.dirname(src_header_file)))
            )

    return records


def filter_success(df):
    """Filter success rows."""
    return df[
        (df["phism_synth"] == "SUCCESS")
        & (
            ((df["tbgen_cosim"] == "SUCCESS") & (df["phism_cosim"] == "SUCCESS"))
            | (df["syn_latency"])
        )
    ]


# ----------------------- Data processing ---------------------------


def expand_resource_field(field):
    """Will turn things like "res_avail" to a list ['DSP_avail', 'FF_avail', ...]"""
    if "res_" not in field:
        return [field]
    avail = field.split("_")[-1]
    return ["{}_{}".format(res, avail) for res in RESOURCE_FIELDS]


def expand_field(field):
    """Turn a nested namedtuple into a flattened one."""
    if "res_" in field:
        return expand_resource_field(field)
    if "run_status" in field:
        return RUN_STATUS_FIELDS
    return [field]


def is_list_record(x):
    return isinstance(x, (Resource, RunStatus))


def flatten_record(record):
    """Flatten a Record object into a list."""
    return list(
        itertools.chain(*[list(x) if is_list_record(x) else [x] for x in record])
    )


def to_pandas(records):
    """From processed records to pandas DataFrame."""
    cols = list(itertools.chain(*[expand_field(field) for field in RECORD_FIELDS]))
    data = list([flatten_record(r) for r in records])
    data.sort(key=lambda x: x[0])

    # NOTE: dtype=object here prevents pandas converting integer to float.
    return pd.DataFrame(data=data, columns=cols, dtype=object)


# ----------------------- Cosim utilities ---------------------------


@dataclass
class ApMemoryInterface:
    name: str
    ports: List[str]

    def get_num_ports(self) -> int:
        return len(set([port[-1] for port in self.ports]))

    def is_read_only(self, port_id: int) -> bool:
        return f"we{port_id}" not in self.ports

    def is_write_only(self, port_id: int) -> bool:
        return f"q{port_id}" not in self.ports

    def is_read_write(self, port_id: int) -> bool:
        return f"q{port_id}" in self.ports and f"d{port_id}" in self.ports


def get_module_parameters(file: str, module_name: str) -> List[str]:
    """Read the module definition into parameter lists."""
    with open(file, "r") as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    start_line = next(i for i, l in enumerate(lines) if f"module {module_name}" in l)
    end_line = next(i for i, l in enumerate(lines) if ");" in l)

    params = (" ".join(line for line in lines[start_line + 1 : end_line])).split(",")
    return [param.strip() for param in params]


def get_memory_interfaces(params: List[str]):
    """Parse memory interfaces from the module params."""
    interfaces = OrderedDict()
    for param in params:
        prefix = param.split("_")[0]
        if prefix not in interfaces:
            interfaces[prefix] = []
        if param.startswith("ap") or "_" not in param:
            continue
        interfaces[prefix].append(param.split("_")[1])

    return [
        ApMemoryInterface(name, ports)
        for name, ports in interfaces.items()
        if "address0" in ports
    ]


@dataclass
class CosimFixStrategy:
    phism_directives: List[str]
    tbgen_directives: List[str]
    phism_mem_interfaces: List[ApMemoryInterface]
    tbgen_mem_interfaces: List[ApMemoryInterface]

    def empty(self):
        return not self.phism_directives and not self.tbgen_directives


def is_read_write_conflict(
    src_mem: ApMemoryInterface, dst_mem: ApMemoryInterface, port_id: int
) -> bool:
    return (src_mem.is_read_only(port_id) and dst_mem.is_write_only(port_id)) or (
        dst_mem.is_read_only(port_id) and src_mem.is_write_only(port_id)
    )


def fix_cosim_kernels(dir: str) -> CosimFixStrategy:
    """Fix issues with co-simulation.
    Returns directives for (source, destination).
    """

    dir = os.path.abspath(dir)  # canonicalize path
    kernel_name = f"kernel_{os.path.basename(dir)}"

    src_proj_dir = os.path.join(dir, "proj", "solution1")
    assert os.path.isdir(src_proj_dir)

    dst_proj_dir = os.path.join(dir, "tb.backup", "solution1")
    assert os.path.isdir(dst_proj_dir)

    src_kernel = os.path.join(src_proj_dir, "syn", "verilog", f"{kernel_name}.v")
    assert os.path.isfile(src_kernel)

    dst_kernel = os.path.join(dst_proj_dir, "syn", "verilog", f"{kernel_name}.v")
    assert os.path.isfile(dst_kernel)

    src_params = get_module_parameters(src_kernel, kernel_name)
    dst_params = get_module_parameters(dst_kernel, kernel_name)

    src_mems = get_memory_interfaces(src_params)
    dst_mems = get_memory_interfaces(dst_params)

    if len(src_mems) != len(dst_mems):
        raise RuntimeError("The number of ap_memory interfaces should be the same.")
    if [mem.name for mem in src_mems] != [mem.name for mem in dst_mems]:
        raise RuntimeError("The name of the interfaces should be the same.")

    # Determine whether we can fix this.
    strategy = CosimFixStrategy(
        phism_directives=[],
        tbgen_directives=[],
        phism_mem_interfaces=src_mems,
        tbgen_mem_interfaces=dst_mems,
    )

    # Iterate every memory interface to see if there is any chance for fixing them.
    for src_mem, dst_mem in zip(src_mems, dst_mems):
        # If any memory interface from the source uses single port, while the target uses dual ports,
        # we will modify the TCL for Phism.
        if src_mem.get_num_ports() == 1 and dst_mem.get_num_ports() == 2:
            strategy.tbgen_directives.append(
                f"set_directive_interface {kernel_name} {dst_mem.name} -mode ap_memory -storage_type ram_1p"
            )
        elif src_mem.get_num_ports() == 2 and dst_mem.get_num_ports() == 1:
            strategy.tbgen_directives.append(
                f"set_directive_interface {kernel_name} {dst_mem.name} -mode ap_memory -storage_type ram_2p"
            )
        elif src_mem.get_num_ports() == dst_mem.get_num_ports():
            num_ports = src_mem.get_num_ports()
            if num_ports == 2:
                # Make sure the dst_mem is 1 write n read.
                # TODO: is this condition enough for detection?
                if src_mem.is_read_only(1) and dst_mem.is_read_write(1):
                    strategy.tbgen_directives.append(
                        f"set_directive_interface {kernel_name} {dst_mem.name} -mode ap_memory -storage_type ram_1wnr"
                    )
                elif (
                    src_mem.is_read_write(0)
                    and src_mem.is_read_write(1)  # Phism is T2P
                    and (
                        dst_mem.is_read_only(0) or dst_mem.is_read_only(1)
                    )  # tbgen is not
                ):
                    strategy.tbgen_directives.append(
                        f"set_directive_interface {kernel_name} {dst_mem.name} -mode ap_memory -storage_type ram_t2p"
                    )
                elif is_read_write_conflict(
                    src_mem, dst_mem, 0
                ) and is_read_write_conflict(src_mem, dst_mem, 1):
                    strategy.tbgen_directives.append(
                        f"set_directive_interface {kernel_name} {dst_mem.name} -mode ap_memory -storage_type ram_1wnr"
                    )

    return strategy


# ----------------------- Benchmark runners ---------------------------


def discover_examples(d: str, examples: Optional[List[str]] = None) -> List[str]:
    """Find examples in the given directory."""
    if not examples:
        examples = POLYBENCH_EXAMPLES

    return sorted(
        [
            root
            for root, _, _ in os.walk(d)
            if os.path.basename(root).lower() in examples
        ]
    )


def get_phism_env():
    """Get the Phism run-time environment."""
    root_dir = get_project_root()

    phism_env = os.environ.copy()
    phism_env["PATH"] = "{}:{}:{}".format(
        os.path.join(root_dir, "llvm", "build", "bin"),
        os.path.join(root_dir, "build", "bin"),
        phism_env["PATH"],
    )
    phism_env["LD_LIBRARY_PATH"] = "{}:{}:{}:{}".format(
        os.path.join(root_dir, "llvm", "build", "lib"),
        os.path.join(
            root_dir,
            "llvm",
            "build",
            "tools",
            "mlir",
            "tools",
            "polymer",
            "pluto",
            "lib",
        ),
        os.path.join(root_dir, "build", "lib"),
        phism_env["LD_LIBRARY_PATH"],
    )

    return phism_env


def get_top_func(src_file):
    """Get top function name."""
    return "kernel_{}".format(os.path.basename(os.path.dirname(src_file))).replace(
        "-", "_"
    )


def get_top_func_param_names(src_file, pb_dir, llvm_dir=None):
    """From the given C file, we try to extract the top function's parameter list.
    This will be useful for Vitis LLVM rewrite."""

    def is_func_decl(item, name):
        return item["kind"] == "FunctionDecl" and item["name"] == name

    top_func = get_top_func(src_file)
    clang_path = "clang"
    if llvm_dir:
        clang_path = os.path.join(llvm_dir, "build", "bin", "clang")

    # Get the corresponding AST in JSON.
    proc = subprocess.Popen(
        [
            clang_path,
            src_file,
            "-Xclang",
            "-ast-dump=json",
            "-fsyntax-only",
            "-I{}".format(os.path.join(pb_dir, "utilities")),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    data = json.loads(proc.stdout.read())

    # First find the top function declaration entry.
    top_func_decls = list(
        filter(functools.partial(is_func_decl, name=top_func), data["inner"])
    )
    assert len(top_func_decls) == 1, "Should be a single declaration for top."
    top_func_decl = top_func_decls[0]

    # Then get all ParmVarDecl.
    parm_var_decls = filter(
        lambda x: x["kind"] == "ParmVarDecl", top_func_decl["inner"]
    )
    return [decl["name"] for decl in parm_var_decls]


PHISM_VITIS_TCL = """
open_project -reset proj
add_files {dummy_src}
set_top {top_func}

open_solution -reset solution1
set_part "zynq"
create_clock -period "100MHz"
config_compile -pipeline_loops 5
{config}

set ::LLVM_CUSTOM_CMD {{$LLVM_CUSTOM_OPT -no-warn {src_file} -o $LLVM_CUSTOM_OUTPUT}}

csynth_design

exit
"""

TBGEN_VITIS_TCL = """
open_project -reset tb
add_files {{{src_dir}/{src_base}.c}} -cflags "-I {src_dir} -I {work_dir}/utilities -D {pb_dataset}_DATASET" -csimflags "-I {src_dir} -I {work_dir}/utilities -D{pb_dataset}_DATASET"
add_files -tb {{{src_dir}/{src_base}.c {work_dir}/utilities/polybench.c}} -cflags "-I {src_dir} -I {work_dir}/utilities -D{pb_dataset}_DATASET" -csimflags "-I {src_dir} -I {work_dir}/utilities -D{pb_dataset}_DATASET"
set_top {top_func}

open_solution -reset solution1
set_part "zynq"
create_clock -period "100MHz"
{config}

csim_design
csynth_design
cosim_design

exit
"""

COSIM_VITIS_TCL = """
open_project tb

open_solution solution1

cosim_design

exit
"""


class PbFlow:
    """Holds all the pb-flow functions.
    TODO: inherits this from PhismFlow.
    """

    def __init__(self, work_dir: str, options: PbFlowOptions):
        """Constructor. `work_dir` is the top of the polybench directory."""
        self.env = get_phism_env()
        self.root_dir = get_project_root()
        self.work_dir = work_dir
        self.cur_file = None
        self.c_source = None
        self.options = options

        self.status = 0
        self.errmsg = "No Error"

    def run(self, src_file):
        """Run the whole pb-flow on the src_file (*.c)."""
        self.cur_file = src_file
        self.c_source = src_file  # Will be useful in some later stages

        # The whole flow
        try:
            (
                self.compile_c()
                .preprocess()
                .split_statements()
                .extract_top_func()
                .polymer_opt()
                .loop_transforms()
                .constant_args()
                .lower_llvm()
                .vitis_opt()
                .run_vitis()
            )
        except Exception as e:
            self.status = 1
            self.errmsg = e

    def run_command(
        self, cmd: str = "", cmd_list: Optional[List[str]] = None, **kwargs
    ):
        """Single entry for running a command."""
        if cmd_list:
            if self.options.dry_run:
                print(" ".join(cmd_list))
                return
            return subprocess.run(cmd_list, **kwargs)
        else:
            if self.options.dry_run:
                print(cmd)
                return
            return subprocess.run(cmd, **kwargs)

    def get_program_abspath(self, program: str) -> str:
        """Get the absolute path of a program."""
        return str(
            subprocess.check_output(["which", program], env=self.env), "utf-8"
        ).strip()

    def compile_c(self):
        """Compile C code to MLIR using mlir-clang."""
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(".c", ".mlir")

        self.run_command(cmd=f'sed -i "s/static//g" {src_file}', shell=True)
        self.run_command(
            cmd_list=[
                self.get_program_abspath("mlir-clang"),
                src_file,
                "-memref-fullrank",
                "-D",
                f"{self.options.dataset}_DATASET",
                "-I={}".format(
                    os.path.join(
                        self.root_dir,
                        "llvm",
                        "build",
                        "lib",
                        "clang",
                        "13.0.0",
                        "include",
                    )
                ),
                "-I={}".format(os.path.join(self.work_dir, "utilities")),
            ],
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )
        return self

    def preprocess(self):
        """Do some preprocessing before extracting the top function."""
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".pre.mlir"
        )
        self.run_command(
            cmd_list=[
                self.get_program_abspath("mlir-opt"),
                src_file,
                "-sccp",
                "-canonicalize",
            ],
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )
        return self

    def split_statements(self):
        """Use Polymer to split statements."""
        if self.options.split == "NO_SPLIT":
            return self

        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", f".{self.options.split.lower()}.mlir"
        )
        log_file = self.cur_file.replace(".mlir", ".log")

        self.run_command(
            cmd_list=[
                self.get_program_abspath("polymer-opt"),
                src_file,
                "-reg2mem",
                (
                    "-annotate-splittable"
                    if self.options.split == "SPLIT"
                    else "-annotate-heuristic"
                ),
                "-scop-stmt-split",
                "-canonicalize",
            ],
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self

    def extract_top_func(self):
        """Extract the top function and all the stuff it calls."""
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".kern.mlir"
        )
        log_file = self.cur_file.replace(".mlir", ".log")
        args = [
            self.get_program_abspath("phism-opt"),
            src_file,
            f'-extract-top-func="name={get_top_func(src_file)}"',
            "-improve-pipelining" if self.options.improve_pipelining else "",
        ]
        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )
        return self

    def polymer_opt(self):
        """Run polymer optimization."""
        if not self.options.polymer:
            return self

        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".plmr.mlir"
        )
        log_file = self.cur_file.replace(".mlir", ".log")

        passes = []
        if self.options.split == "NO_SPLIT":  # The split stmt has applied -reg2mem
            passes += [
                "-reg2mem",
            ]
        passes += [
            "-extract-scop-stmt",
            "-pluto-opt",
        ]

        self.run_command(
            cmd_list=(
                [
                    self.get_program_abspath("polymer-opt"),
                    src_file,
                ]
                + passes
            ),
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self

    def loop_transforms(self):
        """Run Phism loop transforms."""
        if not self.options.loop_transforms:
            return self

        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".loop_transforms.mlir"
        )
        log_file = self.cur_file.replace(".mlir", ".log")

        args = [
            self.get_program_abspath("phism-opt"),
            src_file,
            f'-loop-transforms="max-span={self.options.max_span}"',
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self

    def constant_args(self):
        """Run Phism constant args."""
        if not self.options.constant_args:
            return self

        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".constant_args.mlir"
        )
        log_file = self.cur_file.replace(".mlir", ".log")

        args = [
            self.get_program_abspath("phism-opt"),
            src_file,
            "-replace-constant-arguments",
            "-canonicalize",
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self

    def lower_llvm(self):
        """Lower from MLIR to LLVM."""
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(".mlir", ".llvm")

        args = [
            self.get_program_abspath("mlir-opt"),
            src_file,
            "-lower-affine",
            "-convert-scf-to-std",
            "-canonicalize",
            '-convert-std-to-llvm="use-bare-ptr-memref-call-conv=1"',
            f"| {self.get_program_abspath('mlir-translate')} -mlir-to-llvmir",
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self

    def vitis_opt(self):
        """Optimize LLVM IR for Vitis."""
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".llvm", ".vitis.llvm"
        )

        xln_names = get_top_func_param_names(
            self.c_source, self.work_dir, llvm_dir=os.path.join(self.root_dir, "llvm")
        )

        args = [
            os.path.join(self.root_dir, "llvm", "build", "bin", "opt"),
            src_file,
            "-S",
            "-enable-new-pm=0",
            '-load "{}"'.format(
                os.path.join(self.root_dir, "build", "lib", "VhlsLLVMRewriter.so")
            ),
            "-strip-debug",
            "-mem2arr",
            "-instcombine",
            "-xlnmath",
            "-xlnname",
            "-xlnanno",
            '-xlntop="{}"'.format(get_top_func(src_file)),
            '-xlnnames="{}"'.format(",".join(xln_names)),
            "-strip-attr",
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self

    def run_vitis(self, strategy: Optional[CosimFixStrategy] = None):
        """Run synthesize/testbench generation/co-simulation."""
        src_file = self.cur_file
        base_dir = os.path.dirname(src_file)
        top_func = get_top_func(src_file)

        phism_vitis_tcl = os.path.join(base_dir, "phism.tcl")
        tbgen_vitis_tcl = os.path.join(base_dir, "tbgen.tcl")
        cosim_vitis_tcl = os.path.join(base_dir, "cosim.tcl")

        run_config = "config_bind -effort high"
        if self.options.debug:
            run_config = ""

        # Run Phism Vitis
        dummy_src = src_file.replace(".llvm", ".dummy.c")
        with open(dummy_src, "w") as f:
            f.write("void {}() {{}}".format(top_func))

        # Write the TCL for Phism.
        with open(phism_vitis_tcl, "w") as f:
            phism_run_config = [str(run_config)]
            if strategy:
                phism_run_config.extend(strategy.phism_directives)

            f.write(
                PHISM_VITIS_TCL.format(
                    src_file=src_file,
                    dummy_src=dummy_src,
                    top_func=top_func,
                    config="\n".join(phism_run_config),
                )
            )

        # Write the TCL for TBGEN.
        with open(tbgen_vitis_tcl, "w") as f:
            tbgen_run_config = [str(run_config)]
            if strategy:
                tbgen_run_config.extend(strategy.tbgen_directives)

            f.write(
                TBGEN_VITIS_TCL.format(
                    src_dir=base_dir,
                    src_base=os.path.basename(src_file).split(".")[0],
                    top_func=top_func,
                    work_dir=self.work_dir,
                    config="\n".join(tbgen_run_config),
                    pb_dataset=self.options.dataset,
                )
            )
        with open(cosim_vitis_tcl, "w") as f:
            f.write(COSIM_VITIS_TCL)

        if self.options.dry_run:
            return self

        # Clean up old results
        shutil.rmtree(os.path.join(base_dir, "proj"), ignore_errors=True)
        shutil.rmtree(os.path.join(base_dir, "tb"), ignore_errors=True)
        shutil.rmtree(os.path.join(base_dir, "tb.backup"), ignore_errors=True)

        shutil.rmtree(
            os.path.join(base_dir, "phism.vitis_hls.stdout.log"), ignore_errors=True
        )
        shutil.rmtree(
            os.path.join(base_dir, "tbgen.vitis_hls.stdout.log"), ignore_errors=True
        )
        shutil.rmtree(
            os.path.join(base_dir, "cosim.vitis_hls.stdout.log"), ignore_errors=True
        )

        phism_proc = subprocess.Popen(
            ["vitis_hls", phism_vitis_tcl],
            cwd=base_dir,
            stdout=open(os.path.join(base_dir, "phism.vitis_hls.stdout.log"), "w"),
            stderr=open(os.path.join(base_dir, "phism.vitis_hls.stderr.log"), "w"),
        )

        # Run tbgen Vitis
        if self.options.cosim:
            tbgen_proc = subprocess.Popen(
                ["vitis_hls", tbgen_vitis_tcl],
                cwd=base_dir,
                stdout=open(os.path.join(base_dir, "tbgen.vitis_hls.stdout.log"), "w"),
                stderr=open(os.path.join(base_dir, "tbgen.vitis_hls.stderr.log"), "w"),
            )

            # Allows phism_proc and tbgen_proc run in parallel.
            phism_ret = phism_proc.wait()
            tbgen_ret = tbgen_proc.wait()

            assert phism_ret == 0, "Phism syn failed."
            assert tbgen_ret == 0, "tbgen failed."

            # TODO: add some sanity checks
            phism_syn_verilog_dir = os.path.join(
                base_dir, "proj", "solution1", "syn", "verilog"
            )
            tbgen_syn_verilog_dir = os.path.join(
                base_dir, "tb", "solution1", "syn", "verilog"
            )

            assert os.path.isdir(phism_syn_verilog_dir), "{} doens't exist.".format(
                phism_syn_verilog_dir
            )
            assert os.path.isdir(tbgen_syn_verilog_dir), "{} doens't exist.".format(
                tbgen_syn_verilog_dir
            )

            shutil.copytree(
                os.path.join(base_dir, "tb"),
                os.path.join(base_dir, "tb.backup"),
                dirs_exist_ok=True,
            )

            for f in glob.glob(os.path.join(phism_syn_verilog_dir, "*.*")):
                shutil.copy(f, tbgen_syn_verilog_dir)

            # Run cosim for Phism in the end
            cosim_proc = subprocess.Popen(
                ["vitis_hls", cosim_vitis_tcl],
                cwd=base_dir,
                stdout=open(os.path.join(base_dir, "cosim.vitis_hls.stdout.log"), "w"),
                stderr=open(os.path.join(base_dir, "cosim.vitis_hls.stderr.log"), "w"),
            )
            cosim_ret = cosim_proc.wait()

            if cosim_ret == 1:  # Cosim failed.
                # Try to generate a strategy if it hasn't been assigned.
                if strategy is None:
                    strategy = fix_cosim_kernels(base_dir)
                else:
                    # If not, then there is no recovery.
                    assert False, f"Cosim failed and cannot be fixed by {strategy}."
                # If there is a viable strategy, we would try that.
                if not strategy.empty():
                    print(
                        f">>> {os.path.basename(base_dir)} Attempting to fix cosim issues by {strategy} ..."
                    )
                    return self.run_vitis(strategy=strategy)

                # If not, then there is no recovery.
                assert False, f"Cosim failed and cannot be fixed by {strategy}."
        else:
            phism_ret = phism_proc.wait()
            assert phism_ret == 0, "Phism syn failed."

        return self


def pb_flow_process(d: str, work_dir: str, options: PbFlowOptions):
    """Process a single example."""
    # Make sure the example directory and the work directory are both absolute paths.
    # TODO: make it clear what is d.
    d = os.path.abspath(d)
    work_dir = os.path.abspath(work_dir)

    flow = PbFlow(work_dir, options)
    src_file = os.path.join(d, os.path.basename(d) + ".c")

    start = timer()
    flow.run(src_file)
    end = timer()

    if not options.dry_run:
        print(
            '>>> Finished {:15s} elapsed: {:.6f} secs   Status: {}  Error: "{}"'.format(
                os.path.basename(d), (end - start), flow.status, flow.errmsg
            )
        )


def pb_flow_dump_report(options: PbFlowOptions):
    """Dump report to the work_dir."""
    df = to_pandas(process_pb_flow_result_dir(options.work_dir, options))
    print("\n")
    print(df)
    print("\n")

    df.to_csv(os.path.join(options.work_dir, f"pb-flow.report.{get_timestamp()}.csv"))


def pb_flow_runner(options: PbFlowOptions):
    """Run pb-flow with the provided arguments."""
    assert os.path.isdir(options.pb_dir)

    # Copy all the files from the source pb_dir to a target temporary directory.
    if not options.work_dir:
        options.work_dir = os.path.join(
            get_project_root(), "tmp", "phism", "pb-flow.{}".format(get_timestamp())
        )
    if not os.path.exists(options.work_dir):
        shutil.copytree(options.pb_dir, options.work_dir)

    print(
        ">>> Starting {} jobs (work_dir={}) ...".format(options.job, options.work_dir)
    )

    start = timer()
    with Pool(options.job) as p:
        # TODO: don't pass work_dir as an argument. Reuse it.
        p.map(
            functools.partial(
                pb_flow_process, work_dir=options.work_dir, options=options
            ),
            discover_examples(options.work_dir, examples=options.examples),
        )
    end = timer()
    print("Elapsed time: {:.6f} sec".format(end - start))

    print(">>> Dumping report ... ")
    pb_flow_dump_report(options)
