
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <SDL.h>
#include <stdio.h>
#include <Windows.h>
#include <chrono>
#include <curand_kernel.h>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <helper_functions.h> 
#include <filesystem>
#include <math.h>

//##DOGERAY## by Phil

//settings:
//render dimensions
const int SCREEN_WIDTH = 1280;
const int SCREEN_HEIGHT = 720;
//factor to scale window up for small resoultions
const int upscale = 1;










//create namespace for filesystem
namespace fs = std::filesystem;


 //object struct
typedef struct
{
    int type;
    float3 pos;

    float3 rot;
    //face normals
    float3 norm = { -2, -3, -20 };

    //vertex normals
    float3 n1 = { -2, -3, -20 };
    float3 n2 = { -2, -3, -20 };
    float3 n3 = { -2, -3, -20 };
    //tex coords
    float3 t1 = { 0, 1,0};
    float3 t2 = {0, 0,0};
    float3 t3 = {1,0,0};
    bool smooth = false;
    bool tex = false;
    int mat;
    float3 dim;
    float3 col;
    int texnum = -1;

    float3 addional;

}singleobject;



//BVH node struct
typedef struct
{
    bool active ;
    int id;
    int children[2];

 
   
    int count;

    int under;

    float3 min;
    float3 max;
    bool end;

}bvh;



//CPU only BVH building node helper struct. This is info only used on the cpu for buidling that does not need to be put onto the gpu
typedef struct
{
    int under[10000];
    int count;

}bvhunder;




//setup gpu global variables
int debugnum[1] = { -1 };
__constant__ int edebugnum[1] = { 0 };
__constant__ float backgroundintensity[1] = { 1 };
float nbackgroundintensity[1] = { 1 };
__constant__ int anum[1] = { 0 };
int nanum[1] = { 0 };
int nbvhnumnum[1] = { 0 };
__constant__ int dbvhnumnum[1] = { 0 };




//set up number global variables
int objnum = 10000;
int bvhnum = objnum * 2;
int texnum = 1;
int iter = 0;
int backtex = -1;
//settings global variables
float3 campos = { 0, 0, 2 };
float3 look = { 0, 0, 0 };
float aperturee = 0.01f;
float focus_diste = 3;
 int actualbvhnum =0;
 int max_depthh = 50;
 int samples_per_pixell = 1;
 int fovv = 45;

 //calculate screen size
const int s = SCREEN_WIDTH * SCREEN_HEIGHT;


//image data storage
int outr[s] = { 0 };
int outg[s] = { 0 };
int outb[s] = { 0 };
int noutr[s] = { 0 };
int noutg[s] = { 0 };
int noutb[s] = { 0 };


//cuda starter function 
cudaError_t CudaStarter(int* outputr, int* outputg, int* outputb, bvh* nbvhtree, singleobject* allobjects,cudaTextureObject_t* texarray , int divisor);





//vector helper functions:
#ifndef UNIFIED_MATH_CUDA_H
#define UNIFIED_MATH_CUDA_H
__device__
inline float3 getCrossProduct(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__host__ __device__
inline float3 make3(float a)
{
    return make_float3(a,a,a);
}
__device__
inline float4 getCrossProduct(float4 a, float4 b)
{
    float3 v1 = make_float3(a.x, a.y, a.z);
    float3 v2 = make_float3(b.x, b.y, b.z);
    float3 v3 = make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);

    return make_float4(v3.x, v3.y, v3.z, 0.0f);
}

__host__ __device__
inline float getDotProduct(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
inline float getDotProduct(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ float3 getNormalizedVec(const float3 v)
{
    float invLen = 1.0f / sqrtf(getDotProduct(v, v));
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

__device__ float4 getNormalizedVec(const float4 v)
{
    float invLen = 1.0f / sqrtf(getDotProduct(v, v));
    return make_float4(v.x * invLen, v.y * invLen, v.z * invLen, v.w * invLen);
}

__device__
inline float dot3F4(float4 a, float4 b)
{
    float4 a1 = make_float4(a.x, a.y, a.z, 0.f);
    float4 b1 = make_float4(b.x, b.y, b.z, 0.f);
    return getDotProduct(a1, b1);
}

__host__ __device__
inline float getLength(float3 a)
{
    return sqrtf(getDotProduct(a, a));
}


__device__
inline float getLength(float4 a)
{
    return sqrtf(getDotProduct(a, a));
}
//vector operators
__host__ __device__  float3 operator+(const float3& a, const float3& b) {

    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}
__host__ __device__  float3 operator-(const float3& a, const float3& b) {

    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);

}
__host__ __device__  float3 operator*(const float3& a, const float3& b) {

    return make_float3(a.x *b.x, a.y * b.y, a.z * b.z);

}
__host__ __device__  float3 operator/(const float3& a, const float3& b) {

    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);

}
#endif  // UNIFIED_MATH_CUDA_H

//calculate normal direction
__device__ bool get_face_normal(float3 dir, float3 outward_normal) {
    bool front_face = getDotProduct(dir, outward_normal) < 0;
    return front_face;
}




//bounding box (with distance)
__device__ bool aabb2(float3 o, float3 d, float3 a, float3 b, float &dist) {
    float t_min = 0;
    float t_max = 10000;
    float origin[3] = { o.x,o.y,o.z };
    float direction[3] = { d.x,d.y,d.z };
    float min[3] = { a.x,a.y,a.z };
    float max[3] = { b.x,b.y,b.z };
    for (int a = 0; a < 3; a++) {
        auto invD = 1.0f / direction[a];
        auto t0 = (min[a] - origin[a]) * invD;
        auto t1 = (max[a] - origin[a]) * invD;
        if (invD < 0.0f) {

            float old = t0;
            t0 = t1;

            t1 = old;

        }


        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min)
            return false;
    }

    dist = t_min;

    return true;
}

//triangle hit
__device__ float3 hit_tri(float3 rayOrigin,
    float3 rayVector,
   float3 vertex0, float3 vertex1, float3 vertex2)
{
 
 
        const float EPSILON = 0.0001;
  
        float3 edge1, edge2, h, s, q;
        float a, f, u, v;
        edge1 = vertex1 - vertex0;
        edge2 = vertex2 - vertex0;
        h = getCrossProduct(rayVector, edge2);
        a = getDotProduct(edge1, h);
        if (a > -EPSILON && a < EPSILON)
            return make_float3(-1, 0, 0);   
        f = 1.0 / a;
        s = rayOrigin - vertex0;
        u = f * getDotProduct(s, h);
        if (u < 0.0 || u > 1.0)
            return make_float3(-1, 0, 0);
        q = getCrossProduct(s, edge1);
        v = f * getDotProduct(rayVector, q);
        if (v < 0.0 || u + v > 1.0)
            return make_float3(-1, 0, 0);
   
        float t = f * getDotProduct(edge2, q);
        if (t > EPSILON) // ray intersection
        {

            return make_float3(t, 0, 0);
        }
        else
            return make_float3(-1, 0, 0);

   
}

//sphere hit
__device__ float3 hit_sphere(const float3 center, float radius, float3 origin, float3 dir) {
    float3 off = make3(radius+0.1);
   
        float3 oc = origin - center;
        float a = pow(getLength(dir), 2.0f);
        float half_b = getDotProduct(oc, dir);
        float3 outward_normal = (origin - center) / make3(radius);
        float norm = get_face_normal(dir, outward_normal);
        float c = pow(getLength(oc), 2.0f) - radius * radius;
        float discriminant = half_b * half_b - a * c;
        if (discriminant < 0) {
            return make_float3(-1.0, norm, 0);
        }
        else {
            return make_float3((-half_b - sqrt(discriminant)) / a, norm, 0);
        }
 
}
//get bouding box dimensions for objects
__host__ __device__ bool bounding_box(int obj, float3& min, float3& max, singleobject* b) {



    if (b[obj].type == 0) {

        min = b[obj].pos - make3(b[obj].dim.x);
        max = b[obj].pos + make3(b[obj].dim.x);
    }



    else  if (b[obj].type == 2) {
        float3 v1 = b[obj].pos;
        float3 v2 = b[obj].dim;
        float3 v3 = b[obj].rot;
       

        min = make_float3(fmin(v1.x, fmin(v2.x, v3.x)) - 0.01, fmin(v1.y, fmin(v2.y, v3.y)) - 0.01, fmin(v1.z, fmin(v2.z, v3.z)) - 0.01);
        max = make_float3(fmax(v1.x, fmax(v2.x, v3.x)) + 0.01, fmax(v1.y, fmax(v2.y, v3.y)) + 0.01, fmax(v1.z, fmax(v2.z, v3.z)) + 0.01);


    }




    return true;

}




//function for cacluating boudning box of two objects
 void surrounding_box(float3 amin, float3 amax, float3 bmin, float3 bmax, float3 &min,float3 &max) {
    min = make_float3(fmin(amin.x, bmin.x),
        fmin(amin.y, bmin.y),
        fmin(amin.z, bmin.z));

   max = make_float3(fmax(amax.x, bmax.x),
        fmax(amax.y, bmax.y),
        fmax(amax.z, bmax.z));

   
}

 //calculate bounding box of array
 bool arraybound(float3 &min, float3 &max, int objs[], int len, singleobject* b) {
     if (len == 0) return{ false };

    float3 temp_min = make3(-1);
    float3 temp_max = make3(-1);
    bool first_box = true;

    for (int g = 0; g < len; g++) {
        bounding_box(objs[g], temp_min, temp_max, b);
        if (first_box) {
            min = temp_min;
            max = temp_max;
        }
        else {

            surrounding_box(min, max, temp_min, temp_max, min, max);
        }

       
        first_box = false;
    }

    return true;
}

 //check if bvh node is hit(for preview purposes)
 __device__ float3 bvhhit(float3 origin, float3 dir , bvh* bvhtree) {


    //check if in valid index
    if (edebugnum[0] < 0 || edebugnum[0] >= dbvhnumnum[0]) {

        return make_float3(-1, 0, 0);
    }
    float dist = 1;
    if ( aabb2(origin, dir, bvhtree[abs(edebugnum[0])].min, bvhtree[abs(edebugnum[0])].max,dist)) {


        return  make_float3(dist, 0, 0);



    }
    return make_float3(-1, 0, 0);

}


//hit function for objects
__device__ float3 singlehit(float3 origin, float3 dir, int x, singleobject* b) {


    float mindist = 10000;
    bool isnothit = true;
    float closest = 0;
    float3 dist;

    if (b[x].type == 0) {
        dist = hit_sphere(b[x].pos, b[x].dim.x, origin, dir);

    }
    else  if (b[x].type == 2) {
        dist = hit_tri(origin, dir, b[x].pos, b[x].dim, b[x].rot);

    }

    if (dist.x < mindist && dist.x > -0.0) {
    
        mindist = dist.x;
        closest = x;
     
        isnothit = false;
    }

    if (isnothit) {

        mindist = -1;
    }
    return make_float3(mindist, closest, 0);


}


//overall hit function
__device__ float3 hit(float3 origin, float3 dir, bvh* bvhtree, singleobject* b) {

    //cant dynamically allocate(too big)
    //becouse array cant be resized the fucntion koves along the array using part of it

//maximum number of traversals(Make larger to load huge models at cost of memory and performance)
//let me know how to make resizable arrays within a cuda kernel to remove this

    int tracked[10000];
    tracked[0] = 0;

    //array length
    int num = 1;
    //minimum place in array
    int mini = 0;


    //output
    float3 out = make_float3(10000000, 0, 0);

    //is not hit
    bool oof = true;

    bool* test = new bool[10];

    delete[] test;
 
    //while array is not empty
    while (mini < num) {
    


        //get array length(so it doesnt change)
        int numm = num;

        //for each node in length
        for (int node = mini; node < numm; node++) {
















            //remove node from array
            mini++;


            //get boudning box distance of node
            float dister;
            if (aabb2(origin, dir, bvhtree[tracked[node]].min, bvhtree[tracked[node]].max,dister)) {

                //if distance is less that minumim distance so far
                if (dister < out.x) {
                   //if it is an end node
                    if (bvhtree[tracked[node]].end == true) {
                        //run hit function
                        float3 temp = singlehit(origin, dir, bvhtree[tracked[node]].under, b);
                        //if hit
                        if (temp.x > -0.01 && temp.x < out.x) {
                          
                                out = temp;
                                oof = false;
                               
                            


                        }

                    }
                    else {
                        //add child nodes
                        tracked[num] = bvhtree[tracked[node]].children[0];
                        num++;
                        tracked[num] = bvhtree[tracked[node]].children[1];
                        num++;
                    }



                }
              

                






            }



        }

    }
    if (oof == true) {
        out = make_float3(-1, 0, 0);

    }

    return out;

}


















//random fucntions
__device__ float3 random_in_unit_sphere() {
    while (true) {
        curandState state;
        int tId = threadIdx.x + (blockIdx.x * blockDim.x);

        curand_init((unsigned long long)clock() + tId, 0, 0, &state);

       
        auto p = make_float3((curand_uniform_double(&state) * 2) - 1, (curand_uniform_double(&state) * 2) - 1 , (curand_uniform_double(&state) * 2) - 1);
        if (pow(getLength(p), 2.0f) >= 1) continue;
        return p;
    }
}
__device__ float randy() {
    while (true) {
        curandState state;
        int tId = threadIdx.x + (blockIdx.x * blockDim.x);

        curand_init((unsigned long long)clock() + tId, 0, 0, &state);

        double rand1 = (curand_uniform_double(&state) ) ;
      
       
        return rand1;
    }
}

//functions for mats


__device__ float3 reflect( float3 v, float3 n) {
    return v - make_float3(2.0 * getDotProduct(v, n), 2.0 * getDotProduct(v, n), 2.0 * getDotProduct(v, n)) * n;
}

//clamp function cpu
float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__device__ float3 refract(float3 uv, float3 n, float etai_over_etat) {
    float cos_theta = min(getDotProduct(uv * make3(-1), n), 1.0);
    float3 r_out_perp = make3(etai_over_etat) * (uv + make3(cos_theta) * n);
    float3 r_out_parallel = make3(-sqrt(fabs(1.0 - pow(getLength(r_out_perp), 2.0f)))) * n;
    return r_out_perp + r_out_parallel;
}


__device__ float reflectance(float cosine, float ref_idx) {
    // Use Schlick's approximation for reflectance.
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}


__device__ float distance(float3 p1, float3 p2)
{
    float diffY = p1.y - p2.y;
    float diffX = p1.x - p2.x;
    return sqrt((diffY * diffY) + (diffX * diffX));
}


//function for getting normal of objects
__device__ float3 getnormal(int obj, float3 origin, float3 hitpoint, singleobject* b, float3 dir, float3 &texco) {

    

    if (b[obj].type == 0) {
        return (hitpoint - b[obj].pos) / make3(b[obj].dim.x);

    }


	else  if (b[obj].type == 2) {
		float3 uv;

		float3 vertex0 = b[obj].pos;
		float3 vertex1 = b[obj].dim;
		float3 vertex2 = b[obj].rot;

        //get normal
		float3 v0v1 = vertex1 - vertex0;
		float3 v0v2 = vertex2 - vertex0;
	
		float3 N = getCrossProduct(v0v1, v0v2);



		float3 pvec = getCrossProduct(dir, v0v2);
		float det = getDotProduct(v0v1, pvec);

	


		float invDet = 1 / det;

		float3 tvec = origin - vertex0;
        //get uv
		uv.x = getDotProduct(tvec, pvec) * invDet;


		float3 qvec = getCrossProduct(tvec, v0v1);
		uv.y = getDotProduct(dir, qvec) * invDet;


		uv.z = 1 - uv.x - uv.y;
        //get tex coords
		texco = make3(uv.z) * b[obj].t1 + make3(uv.x) * b[obj].t2 + make3(uv.y) * b[obj].t3;

        //checl if object uses new system
		if (b[obj].norm.z != -20) {
			N = b[obj].norm;



            //check if object is smooth
			if (b[obj].n1.z != -20 && b[obj].smooth) {

				float3 n0 = b[obj].n1;
				float3 n1 = b[obj].n2;
				float3 n2 = b[obj].n3;
				N = make3(uv.z) * n0 + make3(uv.x) * n1 + make3(uv.y) * n2;
			}


		}
		return  getNormalizedVec(N);
	}


	return getNormalizedVec(hitpoint - b[obj].pos);


}

//checker uv function
__device__ float3 checker(float3 uv, float3 p, float3 col1, float3 col2) {
    float u2 = floor(uv.x * 10);
    float v2 = floor(uv.y * 10);
    float yes = u2 + v2;
    if (fmod(yes,(float)2) == 0)
        return col1;
    else
        return col2;
}

//main ray fucntion
__device__ float3 raycolor(float3 origin,float3 dir, int max_depth, bvh* bvhtree, singleobject* b, cudaTextureObject_t* tex, int backtex) {
   
    float3 raydir = dir;
    float3 rayo = origin;
    float3 cur_attenuation = make3(1.0f);
    
    for (int i = 0; i < max_depth; i++) {
      
      
      
        //initialize texure coords
        float3 texco;
        //run hit funciton
        float3 hitoride = hit(rayo, raydir,bvhtree, b);
        //get hit object id
        int g = int(hitoride.y);
        //get distance
        float hit = hitoride.x;

        //if distance is greater than zero (hit)
        if (hit > 0.0) {
            

        
         //get hit point
            float3 hitt = make3(hit);
            float3 hitpoint = rayo + (hitt * raydir);
          
          //calculate normals and direction
            float3 N =  getnormal(g, rayo, hitpoint, b, raydir, texco);

            
            bool inorout = get_face_normal(raydir, N);
     
            N = inorout ? N : N * make3(-1);



            //get defualt color
            float3 ocolor = b[g].col;

            //proccess textures
            if (b[g].texnum >= 0) {
                uchar4 C = tex2D<uchar4>(tex[b[g].texnum], texco.x, -texco.y + 1);
               
                ocolor = make_float3(float(C.x) / 255, float(C.y) / 255, float(C.z) / 255);
            }
            else if (b[g].tex) {

               ocolor =  checker(texco, hitpoint, make3(0.8), b[g].col);

            }
          
        
            if (b[g].mat == 0) {

                //diffuse mat
                float3 target = hitpoint + N;
                    if (b[g].addional.x == 0) {
                        target = target+ random_in_unit_sphere();
                    }
                    else {
                        target = target + getNormalizedVec( random_in_unit_sphere());

                    }


                cur_attenuation = cur_attenuation  * ocolor;

                rayo = hitpoint;
                raydir = getNormalizedVec(target - hitpoint);
               
            }
            else if (b[g].mat == 2) {
                //mirror mat
                
                cur_attenuation = cur_attenuation * ocolor;
                rayo = hitpoint;
                raydir = reflect(getNormalizedVec(raydir), N);

            }
            else if (b[g].mat == 3) {
                //metal mat
                float3 reflected = reflect(getNormalizedVec(raydir), N);
                cur_attenuation = cur_attenuation * ocolor;
                rayo = hitpoint;
                
                raydir = reflected + make3(b[g].addional.y) * random_in_unit_sphere();

            }
          
            else if (b[g].mat == 4) {
                //glass mat

                float ir = b[g].addional.y;

                float refraction_ratio = inorout ? (1.0 / ir) : ir;
                float cos_theta = min(getDotProduct(getNormalizedVec(dir) * make3(-1),N), 1.0);
                float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
                bool cannot_refract = (refraction_ratio * sin_theta) > 1.0;
                float3 reflected;
                if (cannot_refract || reflectance(cos_theta, refraction_ratio) > randy()) {
                    reflected = reflect(getNormalizedVec(dir), N);


                }
                    
                else {

                    reflected = refract(getNormalizedVec(dir), N, refraction_ratio);
                }
                  
                cur_attenuation = cur_attenuation * ocolor;
                rayo = hitpoint;
                raydir = reflected;

            }
            
            else{
                //emmisive mat
                return ocolor * cur_attenuation;
            }
     
          



        }
        else {

        if (backtex > -1) {


            float3 unit_direction = getNormalizedVec(raydir);
         
            float m = 2. * sqrt(pow(unit_direction.x, 2.) + pow(unit_direction.y, 2.) + pow(unit_direction.z + 1., 2.0));
            float3 t = unit_direction / make3(m) + make3(.5);
            t.y = -t.y;

            uchar4 color1 = tex2D<uchar4>(tex[backtex], t.x, -t.y + 1);
            float3 color2 = make_float3(float(color1.x) / 255, float(color1.y) / 255, float(color1.z) / 255);


            return  cur_attenuation * color2 * make_float3(backgroundintensity[0], backgroundintensity[0], backgroundintensity[0]);


        }
              //calculate envoriment
            float3 unit_direction = getNormalizedVec(raydir);
            float t = 0.5 * (unit_direction.y + 1.0);
            float3 c = make_float3((1.0 - t), (1.0 - t), (1.0 - t)) * make_float3(1.0, 1.0, 1.0) + make_float3(t, t, t) * make_float3(0.5, 0.7, 1.0);
            return cur_attenuation * c * make_float3(backgroundintensity[0], backgroundintensity[0], backgroundintensity[0]);

        }
       
        
    }
    //return black
    return make_float3(0.0, 0.0, 0.0);
}




//random function for depth of field
__device__ float3 random_in_unit_disk() {
    while (true) {
        float3 p = make_float3((randy()*2)-1, (randy() * 2) - 1, 0);
        if (pow(getLength(p), 2.0f) >= 1) continue;
        return p;
    }
}

//Main kernel for every pixel
__global__ void Kernel(int* outputr, int* outputg, int* outputb, float* settings, bvh* bvhtree, singleobject* b, cudaTextureObject_t* tex)
{
 


    //get x, y and w
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = x * SCREEN_HEIGHT + y;


    //defualt color
    outputr[w] = 0;
    outputg[w] = 0;
    outputb[w] = 0;


    //get aspect ratio
    float aspect = float(SCREEN_WIDTH / settings[11]) / float(SCREEN_HEIGHT / settings[11]);
  
   
    //get fov and convert to radians
    float fov = settings[8] * M_PI/180;
   
   //calculate heigh and width of viewport
    float viewport_height = 2.0 * tan(fov / 2);
    float viewport_width = aspect * viewport_height;
 
    //get camera position and where to look at
    float3 lookfrom = make_float3(settings[0], settings[1], settings[2]);
    float3 lookat = make_float3(settings[3], settings[4], settings[5]);

    //get aperture and focus distance
    
    float focus_dist = settings[7];

    //define up direction
    float3 vup = make_float3(0, 1, 0);
 

    //calculate look at info 
    float3 wu = getNormalizedVec(lookfrom - lookat) ;
    float3 uu = getNormalizedVec(getCrossProduct(vup, wu));
    float3 vu = getCrossProduct(wu, uu);

    //origin is lookfrom
 

    //calculate viewport
    float3 horizontal = make3(focus_dist) * make3(viewport_width) * uu;
    float3 vertical = make3(focus_dist) * make3(viewport_height) * vu;
    float3 lower_left_corner = lookfrom - horizontal / make3(2) - vertical / make3(2) - make3(focus_dist) * wu ;

      //get lens raduis form apeture
    float lens_radius = settings[6] / 2;


    //color value
    float3 ColorOutput = make_float3(0, 0, 0);

    //for each sample per pixel
    for (int s = 0; s < settings[10]; ++s) {
        //set up cuda random
        curandState state;
        curand_init((unsigned long long)clock() + x, 0, 0, &state);

        float nu = ((float(x) + curand_uniform_double(&state)) / float(SCREEN_WIDTH / settings[11]));
        float nv = (float(y) + curand_uniform_double(&state)) / float(SCREEN_HEIGHT / settings[11]);

        //get ray direction
        float3 rd = make3(lens_radius) * random_in_unit_disk();
        float3 offset = uu * make3(rd.x) + vu * make3(rd.y);
        float3 dir = lower_left_corner + make3(nu) * horizontal + make3(nv) * vertical - lookfrom -offset;

        //calculate ray
        ColorOutput = ColorOutput + raycolor(lookfrom +offset, dir, settings[9],  bvhtree, b,tex, settings[12]);

    }

    //divide by samples per pixel
    float scale = 1.0 / settings[10];
    //output
        outputr[w] = ColorOutput.x * 255*scale;
        outputg[w] = ColorOutput.y * 255 * scale;
        outputb[w] = ColorOutput.z * 255 * scale;


    
    



}


//random funcitons for cpu

 double random_double() {
    // Returns a random real in [0,1).
     srand(GetTickCount());
    return rand() / (RAND_MAX + 1.0);
}









 //get number of objects/tris
 int getnum(std::string File) {


    
     using namespace std;
     string myText;
     ifstream MyReadFile;
     // Read from the text file




     MyReadFile.open(File);








     int line = 0;
     // Use a while loop together with the getline() function to read the file line by line

     if (MyReadFile.is_open()) {

         while (getline(MyReadFile, myText)) {
        
            
           
             if (myText[0] == "/"[0])
                 continue;
             if (myText[0] == "*"[0]) {
               
                 continue;


             }
           //increment number
             line++;
         }
       
      
         // Close the file
         MyReadFile.close();
         return line + 1;
     }
     else {

         SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR,
             "File Error",
             "OOF",
             NULL);
         return 0;
     }

 }

 //get texture id from name
 int gettexnum(std::string query, std::string* texpaths) {
     for (int i = 0; i < texnum; i++)
     {

         std::string tobeq = texpaths[i];
         std::transform(tobeq.begin(), tobeq.end(), tobeq.begin(), ::tolower);
         if (tobeq.find(query) != std::string::npos) {
             return i;
         }
     }
     return -1;
 }

 //read rts fiole
void read(std::string File, singleobject* b, std::string* texpaths) {


    
    using namespace std;
    string myText;
    ifstream MyReadFile;
    // Read from the text file
  
  
  
   
   MyReadFile.open(File);
 
    
    
  
     
    
    
   
    int line = 0;
    // Use a while loop together with the getline() function to read the file line by line

    if (MyReadFile.is_open()) {
       
        while (getline(MyReadFile, myText)) {
            //what colum
            int colum = 0;
            stringstream s_stream(myText); //create string stream from the string

            //ignore comments
            if (myText[0] == "/"[0]) {

                continue;
            }
           //read settings
            if (myText[0] == "*"[0]) {
                while (s_stream.good()) {
                    string substr;
                    getline(s_stream, substr, ','); 

                   
                   //set info
                     if (colum == 1) {

                       campos.x = stof(substr);
                    }
                    else if (colum == 2) {

                        campos.y= stof(substr);
                    }
                    else if (colum == 3) {

                        campos.z = stof(substr);
                    }
                    else if (colum == 4) {

                       aperturee = stof(substr);
                    }
                    else if (colum == 5) {

                       look.x = stof(substr);
                    }
                    else if (colum == 6) {

                        look.y = stof(substr);
                    }
                    else if (colum == 7) {

                        look.z = stof(substr);
                    }
                    else if (colum == 8) {

                       focus_diste   = stof(substr);
                    }
                    else if (colum == 9) {

                        fovv = stoi(substr);
                    }
                    else if (colum == 10) {

                        max_depthh = stoi(substr);
                    }
                    else if (colum == 11) {

                        samples_per_pixell = stoi(substr);
                    }
                    else if (colum == 12) {

                        nbackgroundintensity[0] = stof(substr);
                    }
                    else if (colum == 13) {

                         if (substr != "no") {
                           backtex = gettexnum(substr, texpaths);

                         }
                     }
                  


                    colum++;
                }
                continue;


            }
            while (s_stream.good()) {
                string substr;
                getline(s_stream, substr, ','); //get first string delimited by comma
           
                  //random value 
                if (substr == "r") {
                 
                    double r =random_double();
                    substr = to_string(r);
                  

                }
                 //appl info
                if (colum == 0) {
                   b[line].pos.x = stof(substr);


                }
                else if (colum == 1) {

                    b[line].pos.y = stof(substr);
                }
                else if (colum == 2) {

                    b[line].pos.z = stof(substr);
                }
                else if (colum == 3) {

                    b[line].type = stoi(substr);
                }
                else if (colum == 4) {

                    b[line].col.x = stof(substr);
                }
                else if (colum == 5) {

                    b[line].col.y = stof(substr);
                }
                else if (colum == 6) {

                    b[line].col.z = stof(substr);
                }
                else if (colum == 7) {

                    b[line].addional.y = stof(substr);
                }
                else if (colum == 8) {

                    b[line].addional.x = stof(substr);
                }
                else if (colum == 9) {

                    b[line].dim.x = stof(substr);
                }
                else if (colum == 10) {

                    b[line].dim.y = stof(substr);
                }
                else if (colum == 11) {

                    b[line].dim.z = stof(substr);
                }
                else if (colum == 12) {

                    b[line].mat = stoi(substr);
                } else if (colum == 13) {
                    b[line].rot.x = stof(substr);


                }
                else if (colum == 14) {

                    b[line].rot.y = stof(substr);
                }
                else if (colum == 15) {

                    b[line].rot.z = stof(substr);
                }
                  else if (colum == 16) {
                    b[line].norm.x = stof(substr);


                }
                else if (colum == 17) {

                    b[line].norm.y = stof(substr);
                }
                else if (colum == 18) {

                    b[line].norm.z = stof(substr);
                }

                else if (colum == 19) {
                    b[line].n1.x = stof(substr);


                }
                else if (colum == 20) {

                    b[line].n1.y = stof(substr);
                }
                else if (colum == 21) {

                    b[line].n1.z = stof(substr);
                }

                else if (colum == 22) {
                    b[line].n2.x = stof(substr);


                }
                else if (colum == 23) {

                    b[line].n2.y = stof(substr);
                }
                else if (colum == 24) {

                    b[line].n2.z = stof(substr);
                }

                else if (colum == 25) {
                b[line].n3.x = stof(substr);


                }
                else if (colum == 26) {

                b[line].n3.y = stof(substr);
                }
                else if (colum == 27) {

                b[line].n3.z = stof(substr);
                }




                else if (colum == 28) {
                b[line].t1.x = stof(substr);


                }
                else if (colum == 29) {

                b[line].t1.y = stof(substr);
                }
               
                else if (colum == 30) {
                b[line].t2.x = stof(substr);


                }
                else if (colum == 31) {

                b[line].t2.y = stof(substr);
                }
              
                else if (colum == 32) {
                b[line].t3.x = stof(substr);


                }
                else if (colum == 33) {

                b[line].t3.y = stof(substr);
                }
                else if (colum == 34) {
                int is = stoi(substr);
                if (is == 1) {
                    b[line].smooth = true;

                }

               


                }
                else if (colum == 35) {

                int is = stoi(substr);
                if (is == 1) {
                    b[line].tex = true;

                }
                }
                else if (colum == 36) {

                if (substr != "no") {
                    b[line].texnum = gettexnum(substr, texpaths);

                  }
                }
           







                colum++;
            }
            line++;
        }

        //set object number
        nanum[0] = line+1;
        // Close the file
        MyReadFile.close();
    }
    else {

        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR,
            "File Error",
            "ooof",
            NULL);
    }

}


//Sorts an array and puts other array in same order
void pairsort(float* a, int* b, int n)
{
    std::pair<float, int>* pairt = new std::pair<float, int>[n];

    // Storing the respective array
    // elements in pairs.
    for (int i = 0; i < n; i++)
    {
        pairt[i].first = a[i];
        pairt[i].second = b[i];
    }

    // Sorting the pair array.
    std::sort(pairt, pairt + n);

    // Modifying original arrays
    for (int i = 0; i < n; i++)
    {
        a[i] = pairt[i].first;
        b[i] = pairt[i].second;
    }

    delete[] pairt;
}

//calulates standard devation
float3 calculateSD(float3 data[], int len)
{
    float3 out;
    float sum = 0.0, mean, standardDeviation = 0.0;

    int i;

    for (i = 0; i < len; ++i)
    {
        sum += data[i].x;
    }

    mean = sum / len;

    for (i = 0; i < len; ++i) {
        standardDeviation += pow(data[i].x - mean, 2);

    }
       

    out.x = sqrt(standardDeviation / len);




     sum = 0.0, standardDeviation = 0.0;

   

    for (i = 0; i < len; ++i)
    {
        sum += data[i].y;
    }

    mean = sum / len;

    for (i = 0; i < len; ++i) {

        standardDeviation += pow(data[i].y - mean, 2);
    }
        

    out.y = sqrt(standardDeviation / len);

     sum = 0.0, standardDeviation = 0.0;

   

    for (i = 0; i < len; ++i)
    {
        sum += data[i].z;
    }

    mean = sum / len;

    for (i = 0; i < len; ++i) {
        standardDeviation += pow(data[i].z - mean, 2);

    }
       

    out.z = sqrt(standardDeviation / len);
    return out;
}

//sorts objects based on positon on axis
void sorto(float* output, float3 input[], int size, int* yett) {


    //choose axis with most deviation
    float3 dev = calculateSD(input, size);

    int axis = 0;

    float max = fmaxf(dev.x, fmaxf(dev.y, dev.z));
    if (max == dev.x) {
        axis = 0;

    }
    if (max == dev.y) {
        axis = 1;

    }
    if (max == dev.z) {
        axis = 2;

    }



  


  //copy data

    for (int o = 0; o < size; ++o) {

        if (axis == 0) {
            output[o] = input[o].x;

        }
        if (axis == 1) {
            output[o] = input[o].y;
        }
        if (axis == 2) {
            output[o] = input[o].z;

        }
       

    }
   //sort data
    pairsort(output, yett, size);

}
//split objects in array into two groups for bvh
void split(int input[],int* a, int* b, int num, singleobject* bb) {


    //copy data
    float3 *many = new float3[num];
    int* aoutput = new int[num];
    for (int o = 0; o < num; ++o) {

        many[o] = bb[input[o]].pos;
        aoutput[o] = input[o];
    }

   

    //sorts data
    float* output = new float[num];
    sorto(output, many, num, aoutput);
    
   

    //put data into seperate arrays
    int part1 = num / 2;
 
    for (int o = 0; o < part1; ++o) {

        a[o] = aoutput[o];

    }
    for (int o = part1; o < num; ++o) {

        b[o-part1] = aoutput[o];

    }

    //clean up
    delete[] output;
    delete[] aoutput;
    delete[] many;

}

//bvh building algorittm
void build_bvh(bvh* nbvhtree, singleobject* bb) {

   //array of additional bvh node data only for cpu
    bvhunder* under = new bvhunder[bvhnum];

    //clear bvh
    bvh defualt;
    defualt.active = false;
 
    std::fill_n(nbvhtree, bvhnum, defualt);
    actualbvhnum = 1;

    


    //set up first node (top down)
    //set all avluies execpt children(we will set when we proccess the children)


    //add all remaining objects to first node
    for (int o = 0; o < nanum[0]; ++o) {



        under[0].under[o] = o;

    }

    //set settings
    nbvhtree[0].active = true;
    nbvhtree[0].id = 0;
    nbvhtree[0].count = nanum[0]-1;
    nbvhtree[0].end = false;
    float3 firstmin;
    float3 firstmax;
    arraybound(firstmin, firstmax, under[0].under, nanum[0],bb);
   
    nbvhtree[0].min = firstmin;
    nbvhtree[0].max = firstmax;
   
  //keep track of how many are sorted
    int sorted = nanum[0];




    //for each active node with id==iteration
   //calculate split size 
  //split objects under node into two arrays
   //create two bvh nodes each with the split objects under them
                 //if bvh node has only one object under it mark as end    minus 1 from sorted 
                //calculate bounding boxes
                 //assign new bvh nodes as the children of the current bvh node
                   //add to actual bvh number

    for (int iteration = 0; iteration < 100000; ++iteration) {

        if (sorted <= 0) {

            break;
        }

        //copy amount
        int boi = actualbvhnum;
        //for each node
        for (int node = 0; node < boi; ++node) {
            //if it should be proccesed
            if (nbvhtree[node].active == true && nbvhtree[node].id == iteration && nbvhtree[node].end == false) {
                //calulate size
                int size = nbvhtree[node].count;
                int partition1 = size / 2;
                int partition2 = size - partition1;
          
                //split objects
                int *a = new int[partition1];
                int *b = new int[partition2];

                split(under[node].under, a,  b, size, bb);
                
                //create node 1    
                bvh node1;
                actualbvhnum++;
               


               
                node1.active = true;
                node1.id = iteration+1;
                node1.end = false;
                for (int e = 0; e < partition1; ++e) {
                   under[actualbvhnum - 1].under[e] = a[e];
                }


                node1.count = partition1;

                if (node1.count == 1) {
                    sorted--;
                    node1.under = under[actualbvhnum - 1].under[0];
                    node1.end = true;
            
                }

                //calc bounding boxes
                float3 amin;
                float3 amax;
                arraybound(amin, amax, a, partition1, bb);
                node1.min = amin;
                node1.max = amax;
                nbvhtree[node].children[0] = actualbvhnum-1;
                nbvhtree[actualbvhnum-1] = node1;



            

                //create node 2
                bvh node2;
                actualbvhnum++;




                node2.active = true;
                node2.id = iteration + 1;

                for (int e = 0; e < partition2; ++e) {
                    under[actualbvhnum - 1].under[e] = b[e];
                }

                node2.end = false;
                node2.count = partition2;

                if (node2.count == 1) {
                    sorted--;
                    node2.under = under[actualbvhnum - 1].under[0];
                    node2.end = true;
                 
                }
             
                //calc boudning boxes
                float3 bmin;
                float3 bmax;
                arraybound(bmin, bmax, b, partition2, bb);
                node2.min = bmin;
                node2.max = bmax;
                nbvhtree[node].children[1] = actualbvhnum-1;
                nbvhtree[actualbvhnum-1] = node2;
         

               
                //clean up

                delete[] a;
                delete[] b;
               
               
               
            }




        }
     
        

    }
    delete[] under;
  


}





void readtextures(cudaTextureObject_t* texarray,std::string* texpaths) {


    //load textures and alloacate them
    for (int i = 0; i < texnum; i++)
    {
        unsigned char* hData = NULL;
        unsigned int width, height;
        char* imagePath = strcpy(new char[texpaths[i].length() + 1], texpaths[i].c_str());


        sdkLoadPPM4(imagePath, &hData, &width, &height);


        
     

        unsigned int sizee = width * height * sizeof(uchar4);


        // Allocate array and copy image data
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaArray* cuArray;
        cudaMallocArray(&cuArray,
            &channelDesc,
            width,
            height);
        cudaMemcpyToArray(cuArray,
            0,
            0,
            hData,
            sizee,
            cudaMemcpyHostToDevice);

        cudaTextureObject_t         tex;
        cudaResourceDesc            texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));

        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = cuArray;

        cudaTextureDesc             texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = true;
        texDescr.filterMode = cudaFilterModePoint;
        texDescr.addressMode[0] = cudaAddressModeWrap;
        texDescr.addressMode[1] = cudaAddressModeWrap;
        texDescr.addressMode[2] = cudaAddressModeWrap;
        texDescr.addressMode[3] = cudaAddressModeWrap;
        //  texDescr.readMode = cudaReadModeElementType;

        cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL);



        //add to array
        texarray[i] = tex;
        delete[] imagePath;
    }
   
}


int getppmnum() {
    //get number of ppm textures
    std::string path = fs::current_path().string();
     int i = 0;
    for (const auto& entry : fs::directory_iterator(path)) {
      
        std::string newpath{ entry.path().u8string() };
        if (newpath.find("ppm") != std::string::npos || newpath.find("PPM") != std::string::npos) {
           
            std::cout << "Found: " << entry.path() << std::endl;

            i++;
        }
        

    }
    std::cout << i << " textures total" << std::endl;
    return i;
      
}

void getppmpaths(std::string* things) {
    //add texture paths to array
    std::string path = fs::current_path().string();
    int i = 0;
    for (const auto& entry : fs::directory_iterator(path)) {

        std::string newpath{ entry.path().u8string() };
        if (newpath.find("ppm") != std::string::npos || newpath.find("PPM") != std::string::npos) {

            things[i] = newpath;
                i++;
        }


    }
 
  

}


int main(int argc, char* args[])
{
    //check args so that files can be opened directly
    std::string filename;
    if (argc < 2)
    {
        //use defualt scene
        std::cout << "Opening Default Scene" << std::endl;
        filename = "scene.rts";
    }
    else
    {
        filename = args[1];
        std::cout << "Opening:" << filename << std::endl;
    }
   

    //get number of objects
    objnum = getnum(filename);
    std::cout << objnum << " tris" << std::endl;



    //create object array
    singleobject* allobjects = new singleobject[objnum];
    //get number of textures
    texnum = getppmnum();

    //create texure paths array
    std::string* texpaths = new std::string[texnum];
    //get texture paths
    getppmpaths(texpaths);
    //read rts file
    read(filename, allobjects,texpaths);
    //calulate max number of bvh nodes
    bvhnum = nanum[0]*2;
    //create bvh array
    bvh* nbvhtree = new bvh[bvhnum];

   
    //create texure  array
   cudaTextureObject_t* textures = new cudaTextureObject_t[texnum];
  
   //read textures
   readtextures(textures, texpaths);




   //build bounding volume heirarchy
   //vars with n in front are cpu version that will be loaded onto gpu

   std::cout << "Building BVH.." << std::endl;
    build_bvh(nbvhtree, allobjects);
    std::cout << "Done!" << std::endl;

    std::cout << bvhnum << " nodes total" << std::endl;

    //update global variables
    //this can be moved to the CudaStarter function to have these change over time eg: nbackgroundintensity[0] -= 0.1
    
    nbvhnumnum[0] = bvhnum;
    cudaMemcpyToSymbol(anum, &nanum[0], size_t(1) * sizeof(int), size_t(0), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dbvhnumnum, &nbvhnumnum[0], size_t(1) * sizeof(int), size_t(0), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(edebugnum, &debugnum[0], size_t(1) * sizeof(int), size_t(0), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(backgroundintensity, &nbackgroundintensity[0], size_t(1) * sizeof(float), size_t(0), cudaMemcpyHostToDevice);
    
    SDL_Event event;
   
    //The window we'll be rendering to
    bool quit = false;
    SDL_Window* window = NULL;
    SDL_Renderer* renderer;
    //The surface contained by the window
   
    std::cout << "Opening Window:" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    //Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
    }
    else
    {

        //Create window
        SDL_CreateWindowAndRenderer(SCREEN_WIDTH * upscale, SCREEN_HEIGHT * upscale, 0, &window, &renderer);
        SDL_SetWindowTitle(window,
            "DOGERAY");
        if (window == NULL)
        {
            printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        }
        else
        {
          




            int td;
            while (!quit)
            {


                //td is how much the resolution is being divied by for preview
                td = 1;

              
                //pnum is which preview stage we are on. I decided on 3 in total
                int pnum = 0;

                //start timer
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

                //if first sample render and 1/8 res
                if (iter == 0) {

                    cudaError_t cudaStatus = CudaStarter(outr, outg, outb, nbvhtree, allobjects,textures,8);


                    td = 8;
                    iter++;
                }

                //if second sample render and 1/4 res. We are now on the second preview stage
                else if (iter == 1) {

                    cudaError_t cudaStatus = CudaStarter(outr, outg, outb, nbvhtree, allobjects, textures, 4);
                    td = 4;
                    pnum = 1;

                    iter++;
                }

                //if on third sample render and 1/2 res

                else if (iter == 2) {

                    cudaError_t cudaStatus = CudaStarter(outr, outg, outb, nbvhtree, allobjects, textures, 2);
                    td = 2;
                    pnum = 2;
                    
                    iter++;
                }
                //now we can start the full res render
                else if (iter == 3) {
                    cudaError_t cudaStatus = CudaStarter(outr, outg, outb, nbvhtree, allobjects, textures, 1);

                    pnum = 3;
                    iter++;

                }
                //keep rendering full res
                else{
                    cudaError_t cudaStatus = CudaStarter(noutr, noutg, noutb, nbvhtree, allobjects,textures, 1);
                    //add samples to prev render
                    for (int i = 0; i < s; ++i) {
                        outr[i] += (noutr[i]);
                        outg[i] += (noutg[i]);
                        outb[i] += (noutb[i]);
                  
                    }
                    //number of preview samples
                    pnum = 3;
                    //increase iteration number each time
                    iter++;

                }
               
               


              

              
              



              

















                //display pixels from output
                //divide by td to just proccess rendered pixels
                for (int x = 0; x < SCREEN_WIDTH/td; x++) {
                    for (int y = 0; y < SCREEN_HEIGHT/td; y++) {

                        //calulate w from xa and y
                        int w = x * SCREEN_HEIGHT + y;
                        //set pixel color, clamp, and proccess samples
                        SDL_SetRenderDrawColor(renderer, clamp(outr[w]/(iter-pnum),0,255), clamp(outg[w]/(iter-pnum),0,255), clamp(outb[w]/(iter - pnum),0,255), 255);
                        //uncomment next line for party mode(really trippy)!!!!
                        // SDL_SetRenderDrawColor(renderer, sqrt(outr[w] * iter), sqrt(outg[w] * iter), sqrt(outb[w] * iter), 255);

                        //here things are upscaled to bigger windows for smaller resolutions
                        SDL_RenderDrawPoint(renderer, x * upscale*td, y * upscale*td);
                        if (upscale > 1 || td>1) {


                            for (int u = 0; u < (upscale * td); u++) {
                                SDL_RenderDrawPoint(renderer, x * (upscale * td) + u, y * (upscale*td));
                                for (int b = 0; b < (upscale * td); b++) {
                                    SDL_RenderDrawPoint(renderer, x * (upscale * td) + u, y * (upscale * td) + b);

                                }
                            }

                        }
                     
                  
                      


                     



                  






                    }
                }

                //stop timer
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                //display info
                std::cout << '\r' << "d: " << debugnum[0] << " " << "Time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]  " << 1e+6 / std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " FPS       " << (iter - pnum) * samples_per_pixell << " samples";

             

               
                //get input
                SDL_PollEvent(&event);
                //update window
				SDL_RenderPresent(renderer);


              

                //proccess input
               
				switch (event.type)
				{
				case SDL_QUIT:
                    //handle close
					quit = true;
					SDL_DestroyRenderer(renderer);
					break;


				case SDL_KEYDOWN:

					switch (event.key.keysym.sym) {
					case SDLK_RIGHT:

                        //move camera right
						campos.x += 1;

                        //sameple iteration is reset with movement
                       //for a motion blur effect just dont reset iter with motion 
						iter = 0;
						break;
					case SDLK_LEFT:
						campos.x -= 1;
                        //move camera left
					

						iter = 0;
						break;
					case SDLK_UP:
                        //move camera forward
						campos.z -= 1;
						iter = 0;
						break;
					case SDLK_DOWN:
                        //back
						campos.z += 1;
						iter = 0;
						break;
					case SDLK_w:
                        //up
						campos.y -= 0.5;
						iter = 0;
						break;
					case SDLK_s:
                        //down
						campos.y += 0.5;
						iter = 0;
						break;
					case SDLK_a:
                        //a and d cyle through bvh nodes and displays them with the first matirial in the scene
						debugnum[0]--;
						iter = 0;
						break;
					case SDLK_d:
						debugnum[0]++;
						iter = 0;
						break;
					case SDLK_ESCAPE:
                        //handle exit throug escape
						quit = true;
						SDL_DestroyRenderer(renderer);
						break;

					default:
						break;
					}
				}

















			}



		}
	}
 


   
    //delete dynamic arrays
    delete[] nbvhtree;
    delete[] allobjects;
    delete[] textures;

    //reset gpu
    cudaDeviceReset();
    //close window
    SDL_DestroyWindow(window);

    //Quit SDL subsystems
    SDL_Quit();
    //exit program
    return 0;

}



//this function starts the render kernel
cudaError_t CudaStarter(int* outputr, int* outputg, int* outputb, bvh* nbvhtree, singleobject* allobjects,cudaTextureObject_t* texarray, int divisor)
{











  

   

  
    //set up settings values
    float settings[13] = { campos.x , campos.y,campos.z, look.x,  look.y,  look.z, aperturee ,focus_diste,fovv, max_depthh, samples_per_pixell,divisor, backtex };

    //calculate output size
    int size = SCREEN_WIDTH * SCREEN_HEIGHT;
 

    //placeholder pointers
    float* dev_settings = 0;
    int* dev_outputr = 0;
    int* dev_outputg = 0;
    int* dev_outputb = 0;
    bvh* dev_bvhtree = 0;
    singleobject* dev_allobjects = 0;



    //set up error status
    cudaError_t cudaStatus;
    cudaTextureObject_t* dev_texarray = 0;
 



    // Allocate GPU buffers.
    cudaStatus = cudaMalloc((void**)&dev_outputr, size * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_outputg, size * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_outputb, size * sizeof(int));

    cudaStatus = cudaMalloc((void**)&dev_settings, 13 * sizeof(float));

    cudaStatus = cudaMalloc((void**)&dev_bvhtree, bvhnum * sizeof(bvh));
    cudaStatus = cudaMalloc((void**)&dev_allobjects, objnum * sizeof(singleobject));
    cudaStatus = cudaMalloc((void**)&dev_texarray, texnum * sizeof(cudaTextureObject_t));

    //get device
    int device = -1;
    cudaGetDevice(&device);

    // Copy input vectors from host memory to GPU buffers(and prefetch?)
    cudaStatus = cudaMemcpy(dev_settings, settings, 13 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemPrefetchAsync(dev_settings, 13 * sizeof(float), device, NULL);

    cudaStatus = cudaMemcpy(dev_bvhtree, nbvhtree, bvhnum * sizeof(bvh), cudaMemcpyHostToDevice);
    cudaMemPrefetchAsync(dev_bvhtree, bvhnum * sizeof(bvh), device, NULL);

    cudaStatus = cudaMemcpy(dev_allobjects, allobjects, objnum * sizeof(singleobject), cudaMemcpyHostToDevice);
    cudaMemPrefetchAsync(dev_allobjects, objnum * sizeof(singleobject), device, NULL);


    cudaStatus = cudaMemcpy(dev_texarray, texarray, texnum * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    cudaMemPrefetchAsync(dev_texarray, texnum * sizeof(cudaTextureObject_t), device, NULL);


    //calulate blocks and threads
    //8 seems to work best. Becouse threads need to be evenly divided into block some resolutions may have blank areas near the edges
    dim3 threadsPerBlock(8, 8);

    dim3 numBlocks(SCREEN_WIDTH/divisor / threadsPerBlock.x, SCREEN_HEIGHT/ divisor / threadsPerBlock.y);


    //Start Kernel!
    Kernel << <numBlocks, threadsPerBlock >> > (dev_outputr, dev_outputg, dev_outputb, dev_settings, dev_bvhtree, dev_allobjects,dev_texarray);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    cudaDeviceSynchronize();

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
   

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(outputr, dev_outputr, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(outputg, dev_outputg, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(outputb, dev_outputb, size * sizeof(int), cudaMemcpyDeviceToHost);
   

    //free memory to avoid filling up vram
    cudaFree(dev_settings);
    cudaFree(dev_allobjects);
    cudaFree(dev_bvhtree);
    cudaFree(dev_texarray);

Error:
    //just in case
    cudaFree(dev_outputr);
    cudaFree(dev_outputg);
    cudaFree(dev_outputb);
    cudaFree(dev_settings);


    return cudaStatus;
}

