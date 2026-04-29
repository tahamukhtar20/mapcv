import mercantile
import pytest
from mapcv._mapcv_rs import xy, tile, tiles, xy_bounds

def test_xy() -> None:
    lng_lats = [
        (0.0, 0.0),
        (-122.4194, 37.7749),
        (139.6917, 35.6895),
        (-43.1729, -22.9068),
        (179.999, 85.0),
        (-179.999, -85.0)
    ]
    
    for lng, lat in lng_lats:
        m_x, m_y = mercantile.xy(lng, lat, truncate=False)
        r_x, r_y = xy(lng, lat)
        assert pytest.approx(m_x, abs=1e-5) == r_x
        assert pytest.approx(m_y, abs=1e-5) == r_y

def test_tile() -> None:
    lng_lats = [
        (0.0, 0.0, 0),
        (-122.4194, 37.7749, 14),
        (139.6917, 35.6895, 10),
        (-43.1729, -22.9068, 5),
        (0.0, 85.051129, 2),
        (0.0, -85.051129, 2),
        (200.0, 0.0, 2),
        (-200.0, 0.0, 2),
    ]
    
    for lng, lat, zoom in lng_lats:
        m_tile = mercantile.tile(lng, lat, zoom, truncate=True)
        r_tile = tile(lng, lat, zoom)
        
        assert m_tile.x == r_tile.x
        assert m_tile.y == r_tile.y
        assert m_tile.z == r_tile.z

    zoom = 40
    lng, lat = 12.34, 56.78
    m_tile = mercantile.tile(lng, lat, 32, truncate=False)
    r_tile = tile(lng, lat, zoom)
    assert m_tile.x == r_tile.x
    assert m_tile.y == r_tile.y
    assert r_tile.z == 32

def test_xy_bounds() -> None:
    tile_indices = [
        (0, 0, 0),
        (2621, 6331, 14),
        (907, 404, 10)
    ]
    
    for x, y, z in tile_indices:
        m_bounds = mercantile.xy_bounds(x, y, z)
        r_bounds = xy_bounds(x, y, z)
        
        assert pytest.approx(m_bounds.left, abs=1e-5) == r_bounds.west
        assert pytest.approx(m_bounds.right, abs=1e-5) == r_bounds.east
        assert pytest.approx(m_bounds.bottom, abs=1e-5) == r_bounds.south
        assert pytest.approx(m_bounds.top, abs=1e-5) == r_bounds.north

def test_tiles() -> None:
    bboxes = [
        (-122.5, 37.7, -122.4, 37.8),
        (-0.1, -0.1, 0.1, 0.1),
        (179.0, 0.0, -179.0, 1.0)
    ]
    zooms = [10, 12, 14]
    
    for bbox in bboxes:
        for z in zooms:
            m_tiles = list(mercantile.tiles(*bbox, [z]))
            r_tiles = tiles(*bbox, [z])
            
            assert len(m_tiles) == len(r_tiles)
            
            m_set = {(t.x, t.y, t.z) for t in m_tiles}
            r_set = {(t.x, t.y, t.z) for t in r_tiles}
            assert m_set == r_set

    r_tiles = tiles(-0.1, -0.1, 0.1, 0.1, [40])
    m_tiles = list(mercantile.tiles(-0.1, -0.1, 0.1, 0.1, [32]))
    assert {(t.x, t.y, t.z) for t in r_tiles} == {(t.x, t.y, t.z) for t in m_tiles}

def test_xy_bounds_zoom_clamp() -> None:
    z = 40
    max_index = (1 << 32) - 1
    r_bounds = xy_bounds(max_index, max_index, z)
    m_bounds = mercantile.xy_bounds(max_index, max_index, 32)
    assert pytest.approx(m_bounds.left, abs=1e-5) == r_bounds.west
    assert pytest.approx(m_bounds.right, abs=1e-5) == r_bounds.east
    assert pytest.approx(m_bounds.bottom, abs=1e-5) == r_bounds.south
    assert pytest.approx(m_bounds.top, abs=1e-5) == r_bounds.north
