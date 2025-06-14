def build_and_save_image_index(cache_dir, index_path="image_index.pkl"):
    image_dataset = load_dataset(
        "ARTPARK-IISc/VAANI",
        "images", 
        split="train",
        cache_dir=cache_dir,
        num_proc=8  # ALWAYS same num_proc
    )
    
    print(f"Building index for {len(image_dataset)} images...")
    image_index = {}
    
    for idx, row in enumerate(tqdm(image_dataset, desc="Building index")):
        filename = os.path.basename(row["image"]["path"])
        image_index[filename] = idx
    
    with open(index_path, 'wb') as f:
        pickle.dump(image_index, f)
    
    print(f"Saved {index_path}")
    return image_index



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/home/cis/VaaniCache")
    parser.add_argument("--build_index", action="store_true", 
                       help="Build image index (run once)")
    parser.add_argument("--test_loading", action="store_true",
                       help="Test the dataset by loading a few examples")
    args = parser.parse_args()
    
    index_path = "/home/cis/VaaniCache/image_index.pkl"
    
    if args.build_index:
        build_and_save_image_index(args.cache_dir, index_path)
        return

    AUDIO_CONFIGS = ["Delhi_NewDelhi"]
    #AUDIO_CONFIGS.extend(['WestBengal_Alipurduar', 'WestBengal_CoochBehar', 'WestBengal_DakshinDinajpur', 'WestBengal_Darjeeling', 'WestBengal_Jalpaiguri', 'WestBengal_Jhargram', 'WestBengal_Kolkata', 'WestBengal_Malda', 'WestBengal_North24Parganas', 'WestBengal_PaschimMedinipur', 'WestBengal_Purulia'])
    #AUDIO_CONFIGS.extend(['Bihar_Araria', 'Bihar_Begusarai', 'Bihar_Bhagalpur', 'Bihar_Darbhanga', 'Bihar_EastChamparan', 'Bihar_Gaya', 'Bihar_Gopalganj', 'Bihar_Jahanabad', 'Bihar_Jamui', 'Bihar_Kaimur', 'Bihar_Katihar', 'Bihar_Kishanganj', 'Bihar_Lakhisarai', 'Bihar_Madhepura', 'Bihar_Muzaffarpur', 'Bihar_Patna', 'Bihar_Purnia', 'Bihar_Saharsa', 'Bihar_Samastipur', 'Bihar_Saran', 'Bihar_Sitamarhi', 'Bihar_Supaul', 'Bihar_Vaishali', 'Bihar_WestChamparan'])
    #AUDIO_CONFIGS.extend(['Jharkhand_Deoghar', 'Jharkhand_Garhwa', 'Jharkhand_Jamtara', 'Jharkhand_Palamu', 'Jharkhand_Ranchi', 'Jharkhand_Sahebganj'])
    #AUDIO_CONFIGS.extend(['UttarPradesh_Budaun', 'UttarPradesh_Deoria', 'UttarPradesh_Etah', 'UttarPradesh_Ghazipur', 'UttarPradesh_Gorakhpur', 'UttarPradesh_Hamirpur', 'UttarPradesh_Jalaun', 'UttarPradesh_JyotibaPhuleNagar', 'UttarPradesh_Lalitpur', 'UttarPradesh_Lucknow', 'UttarPradesh_Muzzaffarnagar', 'UttarPradesh_Saharanpur', 'UttarPradesh_Varanasi', 'Uttarakhand_TehriGarhwal', 'Uttarakhand_Uttarkashi'])

    dataset = VAAPairedDataset(AUDIO_CONFIGS, args.cache_dir, index_path)
    
    if args.test_loading:

        print("\nTesting dataset loading...")
        for i, item in enumerate(dataset):
            print(f"\nExample {i}:")
            print(f"  Audio shape: {item['audio'].shape}")
            print(f"  Sampling rate: {item['sampling_rate']}")
            print(f"  Image: {item['image'].shape}")
            
            if i >= 2: 
                break
    

if __name__ == "__main__":
    main()