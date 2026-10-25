"""matplotlib's backend in run_python's sandbox (tools.py), set in sandbox/matplotlibrc.

It draws with Agg, as any headless backend does, but plt.show() keeps each open figure as a PNG instead of
opening a window, and so does the end of the program for the figures still open then. As the program exits,
the PNGs are written after everything it printed: a line with the marker run_python passes in
AGENT_LOOP_FIGURES, then one base64 PNG per line. run_python cuts them off what the model reads, saves them
in the conversation's folder and the page shows them. The container cannot write anywhere run_python could
read, so stdout is the one way out.
"""

import atexit
import base64
import io
import os
import sys

from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backends.backend_agg import FigureCanvasAgg

FigureCanvas = FigureCanvasAgg
FigureManager = FigureManagerBase
MAX_FIGURES = 4  # tools.MAX_FIGURES: a loop that draws a hundred charts sends four
DPI = 100

pngs: list[bytes] = []


def show(*args, **kwargs):
    for manager in Gcf.get_all_fig_managers():
        if len(pngs) < MAX_FIGURES:
            buffer = io.BytesIO()
            manager.canvas.figure.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight")
            pngs.append(buffer.getvalue())
    Gcf.destroy_all()


# Registered after matplotlib's own exit handler (which closes every figure), so it runs before it.
@atexit.register
def send():
    show()
    mark = os.environ.get("AGENT_LOOP_FIGURES")
    if pngs and mark:
        sys.stdout.write(f"\n{mark}\n" + "".join(base64.b64encode(png).decode() + "\n" for png in pngs))
        sys.stdout.flush()
