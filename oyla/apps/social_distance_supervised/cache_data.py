import glob
from getting_input import get_input
class CacheData():
    
    def __init__(self,folder_in, batch_size = 4, flag_upsample=False, frame_start = -1, frame_end = 1000000, min_size = 800, max_size = 1333):

        self.folder_in = folder_in
        
        self.batch_size = batch_size
        self.number_images = len(glob.glob(self.folder_in+'/rgb_*.png'))
        self.flag_upsample = flag_upsample
        self.frame_start = max(frame_start, 0)
        self.frame_end= min(frame_end+1, self.number_images)
        self.min_size = min_size
        self.max_size = max_size

    def caching(self):
        
        number_of_images_in_batch=-1
        rgb_list=[]
        pcd_array_list=[]
        depth_list=[]
        file_number_list = []
        times=0
        image_list = []
        depth_img_list = []
        
        frame_number=0


        # print("number",self.number_images)
        for file_number in range(self.frame_start, self.frame_end):
            image_at_location, dictionary_for_image, depth,pcd_array,depth_img = get_input(self.folder_in,file_number, self.flag_upsample, self.min_size, self.max_size)
            #=======
            #        for file_number in range(170,self.number_images):
            #            image_at_location, dictionary_for_image, depth,pcd_array,depth_img = get_input(self.folder_in,file_number)
            #>>>>>>> Stashed changes
            rgb_list.append(image_at_location)
            image_list.append(dictionary_for_image)
            depth_list.append(depth)
            pcd_array_list.append(pcd_array)
            file_number_list.append(file_number)
            depth_img_list.append(depth_img)
            
            number_of_images_in_batch+=1
            
            input_dict={}
            if number_of_images_in_batch+1== self.batch_size or file_number == self.number_images:
                input_dict['image_list']=image_list
                input_dict['rgb_list']=rgb_list
                input_dict['depth_list']=depth_list
                input_dict['file_number_list']=file_number_list
                input_dict['pcd_array_list']=pcd_array_list
                input_dict['depth_img_list'] = depth_img_list
                yield input_dict
                                 
                                 #             x = threading.Thread(target=thread_funct, args=(im,pcd_array_list))
                
    #             x.start()
                rgb_list=[]
                pcd_array_list=[]
                IJ_list=[]
                depth_list=[]
                file_number_list=[]
                image_list = []
                depth_img_list = []
                frame_number=1
                times+=1
                number_of_images_in_batch=-1

            # if self.number_frames >0 and file_number > self.number_frames:
            #     break
