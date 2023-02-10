class ImageReport:
    
    @staticmethod
    def report_cv2_mat(img, filename="undefined", mode=0):
        print("==== Image Report System ====")
        print("- image name: \t{}".format(filename))
        print("- image dims: \t{}".format(img.shape))
        min_r = img[:,:,0].min()
        max_r = img[:,:,0].max()
        min_g = img[:,:,1].min()
        max_g = img[:,:,1].max()
        min_b = img[:,:,2].min()
        max_b = img[:,:,2].max()
        print("- min/max per channels: \tR:({},{}) G:({},{}) B:({},{})".format(min_r, max_r,min_g, max_g,min_b, max_b))
        