"""Regress the unqualified `analysis` collision with the earlier CA pilot."""
import subprocess
import sys
import unittest
from pathlib import Path


class ImportResolutionTests(unittest.TestCase):
    def test_pilot_modules_survive_source_analysis_preimport(self):
        home = Path(__file__).resolve().parents[2]
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(home)!r}); "
            f"sys.path.insert(0, {str(home / 'cross_task_ca_reweighting')!r}); "
            "import analysis; "
            "assert 'cross_task_ca_reweighting' in analysis.__file__; "
            "from b1_ranking_preserving_correction.analysis import analyze; "
            "from b1_ranking_preserving_correction.report import render_report; "
            "assert analyze.__module__ == 'b1_ranking_preserving_correction.analysis'; "
            "assert render_report.__module__ == 'b1_ranking_preserving_correction.report'"
        )
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
