import pytest
from mapcv._mapcv_rs import PyTileIndex, fetch_tiles
from mapcv.downloader import download_region

def test_fetch_tiles_mock(httpserver):
    httpserver.expect_request("/tile/14/2621/6331.png").respond_with_data(b"FAKE_PNG_1")
    httpserver.expect_request("/tile/14/2622/6331.png").respond_with_data(b"FAKE_PNG_2")
    
    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    
    tiles = [
        PyTileIndex(2621, 6331, 14),
        PyTileIndex(2622, 6331, 14)
    ]
    
    progress_updates = []
    def callback(c: int) -> None:
        progress_updates.append(c)
        
    results = fetch_tiles(tiles, url_template, callback=callback, max_connections=2, policy="strict")
    
    assert len(results) == 2
    assert progress_updates == [1, 2]
    
    results_dict = {(t.x, t.y, t.z): b for t, b in results}
    assert results_dict[(2621, 6331, 14)] == b"FAKE_PNG_1"
    assert results_dict[(2622, 6331, 14)] == b"FAKE_PNG_2"

def test_fetch_tiles_lenient(httpserver):
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(b"FAKE_1")
    httpserver.expect_request("/tile/14/2/1.png").respond_with_data(b"NOT FOUND", status=404)
    
    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    tiles = [
        PyTileIndex(1, 1, 14),
        PyTileIndex(2, 1, 14)
    ]
    
    results = fetch_tiles(tiles, url_template, callback=None, max_connections=2, policy="lenient")
    
    assert len(results) == 1
    assert results[0][0].x == 1

def test_fetch_tiles_strict(httpserver):
    httpserver.expect_request("/tile/14/1/1.png").respond_with_data(b"NOT FOUND", status=404)
    
    url_template = httpserver.url_for("/tile/{z}/{x}/{y}.png")
    tiles = [PyTileIndex(1, 1, 14)]
    
    with pytest.raises(RuntimeError, match="Tile 404 Not Found"):
        fetch_tiles(tiles, url_template, callback=None, max_connections=2, policy="strict")
