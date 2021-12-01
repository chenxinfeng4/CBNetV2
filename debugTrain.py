import sys
import runpy

args = r'python  -m multiview_calib.scripts.compute_relative_poses_robust -s setup.json -i intrinsics_rational.json -l landmarks.json -m lmeds -n 5 -f filenames.json --dump_images'
args = args.split()
if args[0] == 'python':
    """pop up the first in the args"""
    args.pop(0)

if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path


sys.argv.extend(args[1:])

fun(args[0], run_name='__main__')
