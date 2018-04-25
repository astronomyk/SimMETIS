import numpy as np
import simcado as sim
from simcado.spectral import EmissionCurve, TransmissionCurve
from simcado.psf import PSFCube, PSF
from simcado.commands import UserCommands

cmd = sim.UserCommands()
opt = sim.OpticalTrain(cmd)


####################################
## TO-DO List ##
####################################

def test_have_we_implemented_proper_emissivity():
    """
    If we have, then emissivity won't be equal to 0.1
    See line 373 of optics.py
    """
    assert opt.emissiviry != 0.1
    
    
## Tests ###########################



def test_gen_telescope_shake():
    gauss = opt._gen_telescope_shake()
    assert type(gauss == PSF)
    assert gauss.params["pix_res"] == cmd["SCOPE_JITTER_FWHM"]
    
    
def test_gen_master_psf():
    psf = opt._gen_master_psf()
    

def test_optical_train_attributes():
    
    assert isinstance(opt.cmds, UserCommands)
    
    assert isinstance(opt.pix_res, float)
    assert isinstance(opt.psf, PSFCube)
    
    assert isinstance(opt.lam, np.ndarray)
    assert isinstance(opt.lam_res, float)
    assert isinstance(opt.lam_bin_edges, np.ndarray)
    assert isinstance(opt.lam_bin_centers, np.ndarray)
    
    assert isinstance(opt.tc_mirror, TransmissionCurve) or None
    assert isinstance(opt.ec_mirror, EmissionCurve) or None
    assert isinstance(opt.ph_mirror, EmissionCurve) or None
    assert isinstance(opt.n_ph_mirror, float)            
    
    assert isinstance(opt.tc_atmo, TransmissionCurve) or None
    assert isinstance(opt.ec_atmo, EmissionCurve) or None
    assert isinstance(opt.ph_atmo, EmissionCurve) or None
    assert isinstance(opt.n_ph_atmo, float)   
    





    
    
    
