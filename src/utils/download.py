from datasets import load_dataset
import tqdm
import os
import os
# Nuke every progress bar ------------------------------
#os.environ["HF_DATASETS_DISABLE_PROGRESS_BAR"] = "1"   # dataset side
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]   = "1"     # hub/file-download side
#os.environ["HF_DATASETS_CACHE"] = "/home/cis/hieee/.cache/huggingface/"

#from datasets import load_dataset, disable_progress_bar
from huggingface_hub.utils import disable_progress_bars as disable_hub_bars

#disable_progress_bar()      # datasets
disable_hub_bars()          # hub
# ------------------------------------------------------

from datasets import DownloadConfig
dl_cfg = DownloadConfig(disable_tqdm=True)  # belt-and-suspenders

downloaded_txt = open("downloaded.txt", "a")

AUDIO_CONFIGS = ["Delhi_NewDelhi", "images"]
AUDIO_CONFIGS.extend(['WestBengal_Alipurduar', 'WestBengal_CoochBehar', 'WestBengal_DakshinDinajpur', 'WestBengal_Darjeeling', 'WestBengal_Jalpaiguri', 'WestBengal_Kolkata', 'WestBengal_Malda', 'WestBengal_North24Parganas', 'WestBengal_PaschimMedinipur', 'WestBengal_Purulia'])
AUDIO_CONFIGS.extend(['Bihar_Araria', 'Bihar_Begusarai', 'Bihar_Bhagalpur', 'Bihar_Darbhanga', 'Bihar_EastChamparan', 'Bihar_Gaya', 'Bihar_Gopalganj', 'Bihar_Jahanabad', 'Bihar_Jamui', 'Bihar_Kaimur', 'Bihar_Katihar', 'Bihar_Kishanganj', 'Bihar_Lakhisarai', 'Bihar_Madhepura', 'Bihar_Muzaffarpur', 'Bihar_Patna', 'Bihar_Purnia', 'Bihar_Saharsa', 'Bihar_Samastipur', 'Bihar_Saran', 'Bihar_Sitamarhi', 'Bihar_Supaul', 'Bihar_Vaishali', 'Bihar_WestChamparan'])
AUDIO_CONFIGS.extend(['Jharkhand_Deoghar', 'Jharkhand_Garhwa', 'Jharkhand_Jamtara', 'Jharkhand_Palamu', 'Jharkhand_Ranchi', 'Jharkhand_Sahebganj'])
AUDIO_CONFIGS.extend(['UttarPradesh_Budaun', 'UttarPradesh_Deoria', 'UttarPradesh_Etah', 'UttarPradesh_Ghazipur', 'UttarPradesh_Gorakhpur', 'UttarPradesh_Hamirpur', 'UttarPradesh_Jalaun', 'UttarPradesh_JyotibaPhuleNagar', 'UttarPradesh_Lalitpur', 'UttarPradesh_Lucknow', 'UttarPradesh_Muzzaffarnagar', 'UttarPradesh_Saharanpur', 'UttarPradesh_Varanasi', 'Uttarakhand_TehriGarhwal', 'Uttarakhand_Uttarkashi'])

for config in tqdm.tqdm(AUDIO_CONFIGS, desc="Downloading datasets"):   
    ds = load_dataset(
        "ARTPARK-IISc/VAANI",
        config,
        split="train",
        num_proc=8,
        download_config=dl_cfg,
    )
    downloaded_txt.write(f"{config}\n")
    downloaded_txt.flush()
downloaded_txt.close()