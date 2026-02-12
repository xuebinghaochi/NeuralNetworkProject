import numpy as np

def find_max_in_2d_array(array):
    h,w=array.shape
    max_i=[0]
    max_j=[0]
    max_value=array[0,0]
    for i in range(h):
        for j in range(w):
            if(array[i,j]>max_value):
                max_i=[i]
                max_j=[j]
                max_value=array[i,j]
            if(array[i,j]==max_value):
                max_i.append(i)
                max_j.append(j)
    return max_i,max_j


class MaxPool2:
    def iterate_regions(self,image):
        h,w,_=image.shape
        new_h=h//2
        new_w=w//2

        for i in range(new_h):
            for j in range(new_w):
                im_region=image[i*2:(i*2+2),j*2:(j*2+2)]
                yield im_region,i,j

    def forward(self,input):
        self.last_input=input
        h,w,num_filters=input.shape
        output=np.zeros((h//2,w//2,num_filters))
        for im_region,i,j in self.iterate_regions(input):
            output[i,j]=np.amax(im_region,axis=(0,1))
        return output

    def backprop(self,d_L_d_out):
        d_L_d_input=np.zeros(self.last_input.shape)
        for im_region,i,j in self.iterate_regions(self.last_input):
            _,_,num_filters=im_region.shape
            for k in range(num_filters):
                i_sum,j_sum=find_max_in_2d_array(im_region[:,:,k])
                for i2,j2 in zip(i_sum,j_sum):
                    d_L_d_input[i*2 + i2, j*2 + j2,k] = d_L_d_out[i,j,k]
        return d_L_d_input

