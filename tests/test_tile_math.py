import mercantile
import pytest
from mapcv._mapcv_rs import xy, tile, tiles, xy_bounds

def test_xy() -> None:
    # Test random coordinates vs mercantile.xy
    lng_lats = [
        (0.0, 0.0),
        (-122.4194, 37.7749), # SF
        (139.6917, 35.6895), # Tokyo
        (-43.1729, -22.9068), # Rio
        (179.999, 85.0), # Edge
        (-179.999, -85.0) # Edge
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
        (-43.1729, -22.9068, 5)
    ]
    
    for lng, lat, zoom in lng_lats:
        m_tile = mercantile.tile(lng, lat, zoom, truncate=False)
        r_tile = tile(lng, lat, zoom)
        
        assert m_tile.x == r_tile.x
        assert m_tile.y == r_tile.y
        assert m_tile.z == r_tile.z

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
    # west, south, east, north
    bboxes = [
        (-122.5, 37.7, -122.4, 37.8), # SF
        (-0.1, -0.1, 0.1, 0.1), # Equator
        (179.0, 0.0, -179.0, 1.0) # Antimeridian crossing
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
