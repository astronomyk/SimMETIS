# compare zeropoints from Roy vs. simcado
from astropy.io import ascii
from astropy.table import join
from matplotlib import pyplot as plt
import numpy as np

roy=ascii.read("zeropoints_roy.txt",names=["filter_name","lam_c_mu","limit_muJy","inband_flux0_ph_s","inband_flux0_e_s","zeropoint_Jy","bg_ph_s_pix","bg_e_s_pix","id"])
simcado=ascii.read("zeropoints_simcado.txt",names=["filter_name","bg_counts_s_pix","source_counts_s","id"])
j=join(roy,simcado,keys="id",join_type="left")


plt.subplot(211)
k=1

for r in j:
	if(np.isfinite(r["source_counts_s"])):
		ratio=r["source_counts_s"]/r["inband_flux0_ph_s"]
		print(ratio)
		plt.plot(k,ratio,'ks')
		plt.text(k,ratio+0.02,r["filter_name_1"],rotation=90,fontsize=5,va="bottom")
		k+=1

plt.ylabel("Simcado's flux / Roy's flux")
plt.title("Zeropoints [ph/s]")


plt.subplot(212)
k=1

for r in j:
	if(np.isfinite(r["bg_counts_s_pix"])):
		ratio=r["bg_counts_s_pix"]/r["bg_ph_s_pix"]
		print(ratio)
		plt.plot(k,ratio,'ks')
		plt.text(k,ratio+0.02,r["filter_name_1"],rotation=90,fontsize=5,va="bottom")
		k+=1

plt.ylabel("Simcado's flux / Roy's flux")
plt.title("Background fluxes [ph/s/px]")

plt.tight_layout()
plt.savefig("zeropoints_compare.png")
plt.clf()