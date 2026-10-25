# The sandbox the run_python tool runs the model's code in (tools.py): Python 3.12 with numpy and pandas,
# openpyxl so pandas can read .xlsx files, tabulate for DataFrame.to_markdown(), which small models reach
# for to look at a table, and matplotlib, whose charts come back to the page (agent_loop_figures.py), with a
# font for Chinese labels. Build it once, before letting the model write code, and again when this file changes:
#
#     docker build -t agent-loop-sandbox harness/sandbox
#
# Every call starts a new container from this image with no network, a read-only file system, limited
# memory, CPU and processes, as an unprivileged user, and the conversation's files read-only at /data.
FROM python:3.12-slim

# tools.SANDBOX_VERSION: the page asks for a rebuild when an image older than this file is found.
LABEL agent-loop-sandbox.version="2"

# The current CA certificates, and the font matplotlib falls back to for Chinese.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*

# A pip older than 26.2.2 also trusts its own copy of certifi, which still has the compromised eMudhra emSign
# certificates (https://discuss.python.org/t/109576). So the fixed pip comes first, from its wheel by hash, and
# installs everything else.
RUN pip install --no-cache-dir "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"

RUN pip install --no-cache-dir numpy==2.5.4 pandas==3.0.6 openpyxl==3.1.5 tabulate==0.10.0 matplotlib==3.11.2

COPY agent_loop_figures.py /usr/local/lib/python3.12/site-packages/
COPY matplotlibrc /etc/matplotlibrc
ENV MATPLOTLIBRC=/etc/matplotlibrc

WORKDIR /tmp
