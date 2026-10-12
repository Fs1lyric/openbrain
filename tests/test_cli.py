from openbrain.cli import main


def test_cortex_command_reports_rates_and_writes_raster(tmp_path, capsys):
    out = tmp_path / "raster.png"
    assert main(["cortex", "--ms", "200", "--raster", str(out)]) == 0
    assert out.stat().st_size > 0
    assert "Hz" in capsys.readouterr().out


def test_connectome_command_propagates_stimulus(tmp_path, capsys):
    edges = tmp_path / "edges.csv"
    edges.write_text("pre,post,weight\nA,B,1\nB,C,1\n")
    args = ["connectome", str(edges), "--stimulate", "A", "--scale", "30", "--ms", "500"]
    assert main(args) == 0
    out = capsys.readouterr().out
    for name in ("A", "B", "C"):
        assert f"{name} " in out
