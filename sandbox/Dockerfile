# The sandbox the run_python tool runs the model's code in (tools.py): Python 3.12 with numpy and pandas,
# openpyxl so pandas can read .xlsx files, and tabulate for DataFrame.to_markdown(), which small models
# reach for to look at a table. Build it once, before letting the model write code:
#
#     docker build -t agent-loop-sandbox sandbox
#
# Every call starts a new container from this image with no network, a read-only file system, limited
# memory, CPU and processes, as an unprivileged user, and the conversation's files read-only at /data.
FROM python:3.12-slim

RUN pip install --no-cache-dir numpy==2.5.4 pandas==3.0.6 openpyxl==3.1.5 tabulate==0.10.0

WORKDIR /tmp
