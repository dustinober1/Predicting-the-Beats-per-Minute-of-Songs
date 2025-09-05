import tempfile
from pathlib import Path
import subprocess


def test_execute_notebook():
    nb = Path('notebooks/model_comparison.ipynb')
    assert nb.exists(), 'Notebook must exist for this test'

    out_dir = Path(tempfile.mkdtemp())
    out_nb = out_dir / 'model_comparison_executed.ipynb'

    cmd = [
        'jupyter',
        'nbconvert',
        '--to',
        'notebook',
        '--execute',
        str(nb),
        '--ExecutePreprocessor.timeout=120',
        '--output',
        str(out_nb),
    ]

    # Run nbconvert; allow subprocess to raise if fails
    subprocess.check_call(cmd)
    assert out_nb.exists(), 'Executed notebook not produced'
