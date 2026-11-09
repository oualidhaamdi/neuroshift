# NeuroShift — reproducible runtime (SimpleITK backends; FSL is NOT included).
#
#   docker build -t neuroshift .
#   docker run --rm neuroshift                         # runs the test-suite
#   docker run --rm -v "$PWD/RunBatch_Input:/app/RunBatch_Input" \
#                   -v "$PWD/RunBatch_Output:/app/RunBatch_Output" \
#              neuroshift python main.py --register-tissue --analyze --patient <ID>
#
# Without FSL the pipeline automatically falls back to the SimpleITK
# implementations (Otsu brain extraction, K-Means segmentation, Euler3D rigid).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

CMD ["python", "-m", "pytest", "-q"]
