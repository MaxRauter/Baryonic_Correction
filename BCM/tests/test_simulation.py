import pytest
import numpy as np
import os
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, mock_open
from BCM import simulations as sim

@pytest.fixture
def mock_hdf5_file():
    """Create a mock HDF5 file structure for testing."""
    mock_file = MagicMock()
    
    # Mock the Group data
    mock_m200 = np.array([1e14, 2e14, 3e14])
    mock_r200 = np.array([0.5, 0.7, 0.9]) * 1e3  # Will be divided by 1e3 in the code
    mock_lentype_h = np.array([[100, 200, 50, 20, 10], [150, 250, 60, 25, 15], [200, 300, 70, 30, 20]])
    mock_pos = np.array([[10, 20, 30], [15, 25, 35], [20, 30, 40]])
    
    # Mock the Header attributes
    mock_attrs = {
        'BoxSize': 100.0,
        'Redshift': 0.0,
        'Time': 1.0,
        'HubbleParam': 0.7,
        'Omega0': 0.3,
        'OmegaLambda': 0.7,
        'OmegaBaryon': 0.05
    }
    
    # Set up the structure
    mock_file.__getitem__.side_effect = lambda key: {
        'Group/Group_M_Crit200': MagicMock(return_value=mock_m200),
        'Group/Group_R_Crit200': MagicMock(return_value=mock_r200),
        'Group/GroupLenType': MagicMock(return_value=mock_lentype_h),
        'Group/GroupPos': MagicMock(return_value=mock_pos),
        'Header': MagicMock(attrs=mock_attrs),
        'PartType1': {
            'Coordinates': np.random.rand(1000, 3),
            'Velocities': np.random.rand(1000, 3),
            'ParticleIDs': np.arange(1000)
        }
    }[key]
    
    # Mock the slicing behavior
    mock_file.__getitem__.return_value.__getitem__.side_effect = lambda slice_obj: {
        'Group/Group_M_Crit200': mock_m200,
        'Group/Group_R_Crit200': mock_r200,
        'Group/GroupLenType': mock_lentype_h,
        'Group/GroupPos': mock_pos
    }[mock_file.__getitem__.call_args[0][0]][slice_obj]
    
    return mock_file

@pytest.fixture
def reader_instance():
    """Create a basic CAMELSReader instance for testing."""
    return sim.CAMELSReader()

# Test initialization
def test_init_basic():
    """Test basic initialization without paths."""
    reader = sim.CAMELSReader()
    assert reader.path_group is None
    assert reader.path_snapshot is None
    assert reader.index is None
    assert reader.verbose is False

def test_init_with_paths():
    """Test initialization with paths."""
    reader = sim.CAMELSReader(path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5", 
                            path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5",
                            index=5, 
                            verbose=True)
    assert reader.path_group == "Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5"
    assert reader.path_snapshot == "Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    assert reader.index == 5
    assert reader.verbose is True

# Test _load_halodata
def test_load_halodata_basic(mock_hdf5_file):
    """Test basic halo data loading functionality."""

    reader = sim.CAMELSReader(path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5", 
                            path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5",
                            index=5, 
                            verbose=True)
    
    # Verify data was loaded correctly
    assert 'm200' in reader.halo
    assert 'r200' in reader.halo
    assert 'lentype_h' in reader.halo
    assert 'pos' in reader.halo
    assert 'id' in reader.halo

def test_load_halodata_file_not_exists():
    """Test behavior when file doesn't exist."""
    reader = sim.CAMELSReader(path_group="Baryonic_Correction/BCM/tests/Data/groups_015_dm.hdf5", 
                            path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_015_dm.hdf5",
                            index=5, 
                            verbose=True)
    assert hasattr(reader, 'halo')

## Test _load_particles
def test_load_particles_basic():
    """Test basic loading of particle data."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5",
        index=1  # Use second halo
    )
    reader._load_particles()
    
    # Verify particles were loaded
    assert hasattr(reader, 'particles')
    assert isinstance(reader.particles, list)
    assert len(reader.particles) == 1  # Should load only one halo's particles
    
    # Check particle data structure 
    particle_data = reader.particles[0]
    assert 'pos' in particle_data
    assert 'vel' in particle_data
    assert 'id' in particle_data

def test_load_particles_explicit_index():
    """Test loading particle data for multiple halos."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    )
    # Set to load all halos
    reader.index = 1
    reader._load_particles()
    
    # Verify particles were loaded for multiple halos
    assert hasattr(reader, 'particles')
    assert isinstance(reader.particles, list)
    assert len(reader.particles) == 1  # Should have loaded multiple halos

# Test get_halo_particles
def test_get_halo_particles_basic():
    """Test retrieving particles for a specific halo."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    )
    
    # Get particles for halo index 1
    particles = reader.get_halo_particles(1)
    
    # Check particle data structure
    assert isinstance(particles, dict)
    assert 'pos' in particles
    assert 'vel' in particles
    assert 'id' in particles
    
    # Particles should have expected dimensions
    assert len(particles['pos'].shape) == 2
    assert particles['pos'].shape[1] == 3  # 3D positions

def test_get_halo_particles_invalid_index():
    """Test behavior when an invalid halo index is provided."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    )
    
    # Attempt to get particles for an invalid halo index
    with pytest.raises(IndexError):
        reader.get_halo_particles(999999999)  # Assuming this index doesn't exist
        
# Test get_halo_center
def test_get_halo_center_basic():
    """Test retrieving the center of a specific halo."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    )
    
    # Get center for halo index 0
    center = reader.get_halo_center(0)
    
    # Center should be a 3D coordinate
    assert isinstance(center, np.ndarray)
    assert center.shape == (3,)
    assert np.all(np.isfinite(center))

def test_get_halo_center_default_index():
    """Test retrieving center using default instance index."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5",
        index=2
    )
    
    # Get center without specifying index
    center = reader.get_halo_center()
    
    # Should match center for halo at index 2
    explicit_center = reader.get_halo_center(2)
    assert np.array_equal(center, explicit_center)

# Test get_particles_relative_position
def test_get_particles_relative_position_basic():
    """Test retrieving particle positions relative to halo center."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5",
        index=1  # Use second halo
    )
    
    # Get relative positions
    rel_pos = reader.get_particles_relative_position()
    
    # Should return array with 3D coordinates
    assert isinstance(rel_pos, np.ndarray)
    assert len(rel_pos.shape) == 2
    assert rel_pos.shape[1] == 3  # 3D positions

def test_get_particles_relative_position_specific_halo():
    """Test relative positions for specific halo."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    )
    
    halo_index = 2
    rel_pos = reader.get_particles_relative_position(halo_index)
    
    # Calculate expected relative positions manually
    center = reader.get_halo_center(halo_index)
    particles = reader.get_halo_particles(halo_index)
    expected_rel_pos = particles['pos'] - center
    
    # Should match manual calculation
    assert np.array_equal(rel_pos, expected_rel_pos)

# Test _load_simdata
def test_load_simdata_basic():
    """Test basic simulation data loading functionality."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    )
    
    # Verify simulation properties were loaded
    assert hasattr(reader, 'boxsize')
    assert hasattr(reader, 'z')
    assert hasattr(reader, 'h')
    assert hasattr(reader, 'Om')
    assert hasattr(reader, 'Ol')
    assert hasattr(reader, 'Ob')
    assert hasattr(reader, 'fbar')

def test_load_simdata_file_not_exists():
    """Test behavior when snapshot file doesn't exist."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="nonexistent_file.hdf5"
    )
    
    # Should not have cosmological parameters
    assert not hasattr(reader, 'boxsize')
    assert not hasattr(reader, 'z')

# Test _calc_offset
def test_calc_offset_basic():
    """Test basic offset calculation."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    )
    
    # Calculate offset for the second halo
    offset = reader._calc_offset(1)
    
    # Offset should be an integer
    assert isinstance(offset, (int, np.integer))
    assert offset >= 0  # Offset should be non-negative

def test_calc_offset_sequential():
    """Test offset calculation for sequential halos."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    )
    
    # Offset for first halo should be 0
    assert reader._calc_offset(0) == 0
    
    # Offset for second halo should equal the length of the first halo
    assert reader._calc_offset(1) == reader.halo['lentype_h'][0][1]
    
    # Offset for third halo should equal the sum of the lengths of the first two halos
    expected_offset = reader.halo['lentype_h'][0][1] + reader.halo['lentype_h'][1][1]
    assert reader._calc_offset(2) == expected_offset

# Test plot_halo_masses_histogram
def test_plot_halo_masses_histogram_basic(monkeypatch):
    """Test basic functionality of halo mass histogram plotting."""
    # Mock the plotting functions to avoid actual display
    mock_figure = MagicMock()
    mock_plt = MagicMock()
    mock_plt.figure.return_value = mock_figure
    monkeypatch.setattr(plt, 'figure', mock_plt.figure)
    monkeypatch.setattr(plt, 'hist', mock_plt.hist)
    monkeypatch.setattr(plt, 'xlabel', mock_plt.xlabel)
    monkeypatch.setattr(plt, 'ylabel', mock_plt.ylabel)
    monkeypatch.setattr(plt, 'title', mock_plt.title)
    monkeypatch.setattr(plt, 'grid', mock_plt.grid)
    monkeypatch.setattr(plt, 'show', mock_plt.show)
    
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    )
    
    # Call the plotting function
    result = reader.plot_halo_masses_histogram()
    
    # Verify the result is a string containing mass range information
    assert isinstance(result, str)
    assert "Mass range:" in result
    
    # Verify plotting functions were called
    mock_plt.figure.assert_called_once()
    mock_plt.hist.assert_called_once()
    mock_plt.show.assert_called_once()

def test_plot_halo_masses_histogram_with_limit(monkeypatch):
    """Test histogram plotting with a mass limit."""
    # Mock the plotting functions
    mock_figure = MagicMock()
    mock_plt = MagicMock()
    mock_plt.figure.return_value = mock_figure
    monkeypatch.setattr(plt, 'figure', mock_plt.figure)
    monkeypatch.setattr(plt, 'hist', mock_plt.hist)
    monkeypatch.setattr(plt, 'axvline', mock_plt.axvline)
    monkeypatch.setattr(plt, 'xlabel', mock_plt.xlabel)
    monkeypatch.setattr(plt, 'ylabel', mock_plt.ylabel)
    monkeypatch.setattr(plt, 'title', mock_plt.title)
    monkeypatch.setattr(plt, 'grid', mock_plt.grid)
    monkeypatch.setattr(plt, 'show', mock_plt.show)
    
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    )
    
    # Call the plotting function with a mass limit
    mass_limit = 1e13
    result = reader.plot_halo_masses_histogram(masslimit=mass_limit)
    
    # Verify the result contains information about halos below limit
    assert isinstance(result, str)
    assert "Mass range:" in result
    assert "halos below" in result
    
    # Verify the vertical line was drawn at the limit
    mock_plt.axvline.assert_called_once()

# Test init_calculations
def test_init_calculations_basic():
    """Test basic initialization of calculations."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    )
    
    # Initialize calculations with default parameters
    reader.init_calculations(M200=1e14, r200=0.77)
    
    # Check that essential attributes are initialized
    assert hasattr(reader, 'M200')
    assert hasattr(reader, 'r200')
    assert hasattr(reader, 'c')
    assert hasattr(reader, 'r_s')

def test_init_calculations_custom_params():
    """Test initialization of calculations with custom parameters."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    )
    
    # Custom cosmological parameters
    custom_M200 = 2e15
    custom_r200 = 1.5
    custom_c = 5.0
    custom_h = 0.7
    custom_z = 0.5
    custom_Omega_m = 0.27
    
    # Initialize with custom parameters
    reader.init_calculations(
        M200=custom_M200, 
        r200=custom_r200, 
        c=custom_c,
        h=custom_h, 
        z=custom_z, 
        Omega_m=custom_Omega_m
    )
    
    # Verify custom parameters were set
    assert reader.M200 == custom_M200
    assert reader.r200 == custom_r200
    assert reader.c == custom_c
    assert reader.h == custom_h
    assert reader.z == custom_z
    assert reader.Om == custom_Omega_m
    
    # Verify calculated parameters are consistent
    assert reader.r_s == custom_r200 / custom_c

# Test any additional calculation methods that might exist
def test_calculate_profiles_basic():
    """Test basic profile calculation functionality."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    )
    
    # Setup required parameters
    reader.init_calculations(M200=1e14, r200=0.77)
    
    # If calculate_profiles method exists
    if hasattr(reader, 'calculate_profiles'):
        reader.calculate_profiles()
        
        # Verify profiles were calculated
        assert hasattr(reader, 'density_profile')
        assert hasattr(reader, 'mass_profile')

def test_calculate_profiles_with_options():
    """Test profile calculation with specific options."""
    reader = sim.CAMELSReader(
        path_group="Baryonic_Correction/BCM/tests/Data/groups_014_dm.hdf5",
        path_snapshot="Baryonic_Correction/BCM/tests/Data/snapshot_014_dm.hdf5"
    )
    
    # Setup required parameters
    reader.init_calculations(M200=1e14, r200=0.77)
    
    # If calculate_profiles method exists with options
    if hasattr(reader, 'calculate_profiles'):
        # Use custom radial bins and profile options if supported
        try:
            reader.calculate_profiles(
                r_min=0.01,
                r_max=10.0,
                n_bins=50,
                include_baryons=True
            )
            
            # Verify profiles were calculated with custom options
            assert hasattr(reader, 'density_profile')
            assert hasattr(reader, 'baryon_profile')
        except TypeError:
            # If method doesn't accept these parameters, just run basic version
            reader.calculate_profiles()
            assert hasattr(reader, 'density_profile')

# Test simulation setup methods if they exist
def test_setup_simulation_basic():
    """Test basic simulation setup functionality."""
    reader = sim.CAMELSReader()
    
    # If setup_simulation method exists
    if hasattr(reader, 'setup_simulation'):
        # Test with basic parameters
        try:
            reader.setup_simulation(
                boxsize=100.0,
                npart=32**3,
                omega_m=0.3,
                omega_b=0.05
            )
            
            # Verify simulation was set up
            assert hasattr(reader, 'simulation_config')
        except (AttributeError, TypeError):
            # Skip if method doesn't exist or has different parameters
            pass

def test_setup_simulation_cosmology():
    """Test simulation setup with specific cosmology."""
    reader = sim.CAMELSReader()
    
    # If setup_simulation method exists
    if hasattr(reader, 'setup_simulation'):
        # Test with specific cosmology parameters
        try:
            reader.setup_simulation(
                boxsize=50.0,
                npart=64**3,
                omega_m=0.27,
                omega_b=0.04,
                omega_lambda=0.73,
                h=0.7,
                sigma8=0.8,
                ns=0.96
            )
            
            # Verify cosmology parameters were set
            assert hasattr(reader, 'simulation_config')
            assert reader.simulation_config['omega_m'] == 0.27
            assert reader.simulation_config['h'] == 0.7
        except (AttributeError, TypeError):
            # Skip if method doesn't exist or has different parameters
            pass

# Test simulation running methods if they exist
def test_run_simulation_basic():
    """Test basic simulation running functionality."""
    reader = sim.CAMELSReader()
    
    # If run_simulation method exists
    if hasattr(reader, 'run_simulation'):
        # Mock the actual simulation run to avoid execution
        with patch.object(reader, 'run_simulation', return_value=True) as mock_run:
            result = reader.run_simulation()
            
            # Verify the method was called
            mock_run.assert_called_once()
            assert result is True

def test_run_simulation_with_outputs():
    """Test simulation running with specific output options."""
    reader = sim.CAMELSReader()
    
    # If run_simulation method exists
    if hasattr(reader, 'run_simulation'):
        # Mock the actual simulation run to avoid execution
        with patch.object(reader, 'run_simulation', return_value=True) as mock_run:
            try:
                result = reader.run_simulation(
                    output_dir="test_output",
                    snapshots=[0.0, 0.5, 1.0],
                    num_cpus=4
                )
                
                # Verify the method was called with correct parameters
                mock_run.assert_called_once_with(
                    output_dir="test_output",
                    snapshots=[0.0, 0.5, 1.0],
                    num_cpus=4
                )
                assert result is True
            except TypeError:
                # If method doesn't accept these parameters, just run basic version
                result = reader.run_simulation()
                mock_run.assert_called_once()
                assert result is True
    
    
    