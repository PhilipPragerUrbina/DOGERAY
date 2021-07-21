
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

namespace fs = std::filesystem;


//settings
//render dimensions
const int SCREEN_WIDTH = 1280;
const int SCREEN_HEIGHT = 720;

//factor to scale pixels up for small resoultions
const int upscale = 1;






int debugnum[1] = { -1 };

__constant__ int edebugnum[1] = { 0 };
 int objnum = 10000;
 int texnum = 1;


//scene is now stored in scene.rts (ray traced scen file)
//every line is an object
//every attribute is sperated by comma(no spaces)
//there are thirteen attributes(in order)
// use a slash to comment eg:  /hello

// x,y,z,type, r,g,b,extra, lam, dimx,dimy,dimz,mat
//for cubes or models you can also add in rotation x,y,z at the end in degrees
//for traingles position, dimesnn and rotation are each one vertex
//use r isntead of number for random number bewteen 0 and 1
//first three are postion: eg: 0,0,-1
//then there is object type. 0=sphere 1=plane 3=tri
//the next three are rgb values for the matirials.  should be 0-1 but you can go above for lights
//next is extra value. 0-1.  Is for fuzzyness when using metal and for refraction index when using glass
//next is lambert. 0 for no 1 for yes. only for diffuse
//next three are dimensions. For spheres only the x is needed. For planes x and Y are needed. ignore z for now.
//lasty spcify what matirial you want.  0=diffuse  1=light   2=chrome 3=metal 4=glass
 //normals and tex coords are in pairs of 3 eg: 1,2,4
 //then come face normals, 3 vertex nromals, 3 text coords, is smooth, is checke tex, color texture name("no" if none)
//your done now load up the program and it should work
// put settings with *
  // *,camx,camy,camz,apeture, lookx,looky,lookz,focus dist, fov, max depth, samples per frame, background intesnisty 
//end settings




typedef struct
{
    int type;
    float3 pos;

    float3 rot;
    float3 norm = { -2, -3, -20 };


    float3 n1 = { -2, -3, -20 };
    float3 n2 = { -2, -3, -20 };
    float3 n3 = { -2, -3, -20 };
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

__constant__ float backgroundintensity[1] = { 1 };
float nbackgroundintensity[1] = { 1 };


float aperturee = 0.01;
float focus_diste = 3;


float3 campos = { 0, 0, 2 };
float3 look = { 0, 0, 0 };
int max_depthh = 50;
int samples_per_pixell = 1;
int fovv = 45;

__constant__ int anum[1] = { 0 };
int nanum[1] = { 0 };
/*
 float3 npos[objnum] = { {0,0,-1} ,{0,5.4,-1},{0,-1,-1},{1,0,-1},{-1.3,0,-2},{1,0,0},{1,0,0} };
int nobjects[objnum] = { 0,0,0,0,1,0,2 };
 float3 ncol[objnum] = { {0.5, 0.5, 0.5}, { 0.5, 0.5, 0.5 }, { 1, 0, 0 }, { 0.5, 0.5, 0.5 }, { 0, 0, 5 },{ 1, 1, 1 },{ 1, 0, 0 } };
 float3 naddional[objnum] = { {0, 0.5, 0}, {0, 0.5,0 }, { 0, 0.5, 0 }, { 0, 0.2, 0 }, { 0, 0, 0},{ 0,1.5, 0},{ 0,1.5, 0} };

 float3 ndim[objnum] = { {0.5,0,0},{5,0,0},{0.5,0,0},{0.4,0,0},{0.5,0.5,0},{0.3,0,0},{0.3,0,0} };
  int nmats[objnum] = { 0,0,1,3, 1,4,1 };

0,0,-1,0, 0.5,0.5,0.5,0, 0,0.5,0,0,0
0,5.4,-1,0, 0.5,0.5,0.5,0, 0,{5,0,0,0
0,-1,-1,0, 1,0,0,0, 0, 0.5,0,0,1
1,0,-1,0, 0.5,0.5,0.5,0.2, 0,0.4,0,0,3
-1.3,0,-2,1, 0,0,5,0, 0, 0.5,0.5,0,1
1,0,0,0, 1,1,1,1.5, 0,0.3,0,0,4
1,0,0,2, 1,0,0,0, 0,0.3,0,0,1
 
0,0,-1,0,0.5,0.5,0.5,0,0,0.5,0,0,0
0,5.4,-1,0,0.5,0.5,0.5,0,0,{5,0,0,0
0,-1,-1,0,1,0,0,0,0,0.5,0,0,1
1,0,-1,0,0.5,0.5,0.5,0.2,0,0.4,0,0,3
-1.3,0,-2,1,0,0,5,0,0,0.5,0.5,0,1
1,0,0,0,1,1,1,1.5,0,0.3,0,0,4
1,0,0,2,1,0,0,0,0,0.3,0,0,1

*/

typedef struct
{
    bool active ;
    int id;
    int children[2];

    //couasing slowdown!!!!! Dnyamically allocate !!!!!!
   
    int count;

    int under;

    float3 min;
    float3 max;
    bool end;

}bvh;

typedef struct
{
    int under[10000];
    int count;

}bvhunder;


 int actualbvhnum =0;



const int s = SCREEN_WIDTH * SCREEN_HEIGHT;



int iter = 0;
int outr[s] = { 0 };
int outg[s] = { 0 };
int outb[s] = { 0 };
int noutr[s] = { 0 };
int noutg[s] = { 0 };
int noutb[s] = { 0 };
//objects
cudaError_t CudaStarter(int* outputr, int* outputg, int* outputb, bvh* nbvhtree, singleobject* allobjects,cudaTextureObject_t* texarray , int divisor);


  int bvhnum = objnum * 2;
  int nbvhnumnum[1] = { 0 };
  __constant__ int dbvhnumnum[1] = { 0 };



#ifndef UNIFIED_MATH_CUDA_H
#define UNIFIED_MATH_CUDA_H

/*****************************************
                Vector
/*****************************************/

__device__
inline float fastDiv(float numerator, float denominator)
{
    return __fdividef(numerator, denominator);
    //        return numerator/denominator;        
}

__device__
inline float getSqrtf(float f2)
{
    return sqrtf(f2);
    //        return sqrt(f2);
}

__device__
inline float getReverseSqrt(float f2)
{
    return rsqrtf(f2);
}

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

/*****************************************
                Matrix3x3
/*****************************************/
typedef struct
{
    float4 m_row[3];
}Matrix3x3_d;

__device__
inline void setZero(Matrix3x3_d& m)
{
    m.m_row[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m.m_row[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m.m_row[2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__device__
inline Matrix3x3_d getZeroMatrix3x3()
{
    Matrix3x3_d m;
    m.m_row[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m.m_row[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m.m_row[2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return m;
}

__device__
inline void setIdentity(Matrix3x3_d& m)
{
    m.m_row[0] = make_float4(1, 0, 0, 0);
    m.m_row[1] = make_float4(0, 1, 0, 0);
    m.m_row[2] = make_float4(0, 0, 1, 0);
}

__device__
inline Matrix3x3_d getIdentityMatrix3x3()
{
    Matrix3x3_d m;
    m.m_row[0] = make_float4(1, 0, 0, 0);
    m.m_row[1] = make_float4(0, 1, 0, 0);
    m.m_row[2] = make_float4(0, 0, 1, 0);
    return m;
}

__device__
inline Matrix3x3_d getTranspose(const Matrix3x3_d m)
{
    Matrix3x3_d out;
    out.m_row[0] = make_float4(m.m_row[0].x, m.m_row[1].x, m.m_row[2].x, 0.f);
    out.m_row[1] = make_float4(m.m_row[0].y, m.m_row[1].y, m.m_row[2].y, 0.f);
    out.m_row[2] = make_float4(m.m_row[0].z, m.m_row[1].z, m.m_row[2].z, 0.f);
    return out;
}

__device__
inline Matrix3x3_d MatrixMul(Matrix3x3_d& a, Matrix3x3_d& b)
{
    Matrix3x3_d transB = getTranspose(b);
    Matrix3x3_d ans;
    //        why this doesn't run when 0ing in the for{}
    a.m_row[0].w = 0.f;
    a.m_row[1].w = 0.f;
    a.m_row[2].w = 0.f;
    for (int i = 0; i < 3; i++)
    {
        //        a.m_row[i].w = 0.f;
        ans.m_row[i].x = dot3F4(a.m_row[i], transB.m_row[0]);
        ans.m_row[i].y = dot3F4(a.m_row[i], transB.m_row[1]);
        ans.m_row[i].z = dot3F4(a.m_row[i], transB.m_row[2]);
        ans.m_row[i].w = 0.f;
    }
    return ans;
}

/*****************************************
                Quaternion
/*****************************************/

typedef float4 Quaternion;

__device__
inline Quaternion quaternionMul(Quaternion a, Quaternion b);

__device__
inline Quaternion qtNormalize(Quaternion in);

__device__
inline float4 qtRotate(Quaternion q, float4 vec);

__device__
inline Quaternion qtInvert(Quaternion q);

__device__
inline Matrix3x3_d qtGetRotationMatrix(Quaternion q);

__device__
inline Quaternion quaternionMul(Quaternion a, Quaternion b)
{
    Quaternion ans;
    ans = getCrossProduct(a, b);
    ans = make_float4(ans.x + a.w * b.x + b.w * a.x + b.w * a.y, ans.y + a.w * b.y + b.w * a.z, ans.z + a.w * b.z, ans.w + a.w * b.w + b.w * a.w);
    //        ans.w = a.w*b.w - (a.x*b.x+a.y*b.y+a.z*b.z);
    ans.w = a.w * b.w - dot3F4(a, b);
    return ans;
}

__device__
inline Quaternion qtNormalize(Quaternion in)
{
    return getNormalizedVec(in);
    //        in /= length( in );
    //        return in;
}

__device__
inline Quaternion qtInvert(const Quaternion q)
{
    return make_float4(-q.x, -q.y, -q.z, q.w);
}

__device__
inline float4 qtRotate(const Quaternion q, const float4 vec)
{
    Quaternion qInv = qtInvert(q);
    float4 vcpy = vec;
    vcpy.w = 0.f;
    float4 out = quaternionMul(quaternionMul(q, vcpy), qInv);
    return out;
}

__device__
inline float4 qtInvRotate(const Quaternion q, const float4 vec)
{
    return qtRotate(qtInvert(q), vec);
}

__device__
inline Matrix3x3_d qtGetRotationMatrix(Quaternion quat)
{
    float4 quat2 = make_float4(quat.x * quat.x, quat.y * quat.y, quat.z * quat.z, 0.f);
    Matrix3x3_d out;

    out.m_row[0].x = 1 - 2 * quat2.y - 2 * quat2.z;
    out.m_row[0].y = 2 * quat.x * quat.y - 2 * quat.w * quat.z;
    out.m_row[0].z = 2 * quat.x * quat.z + 2 * quat.w * quat.y;
    out.m_row[0].w = 0.f;

    out.m_row[1].x = 2 * quat.x * quat.y + 2 * quat.w * quat.z;
    out.m_row[1].y = 1 - 2 * quat2.x - 2 * quat2.z;
    out.m_row[1].z = 2 * quat.y * quat.z - 2 * quat.w * quat.x;
    out.m_row[1].w = 0.f;

    out.m_row[2].x = 2 * quat.x * quat.z - 2 * quat.w * quat.y;
    out.m_row[2].y = 2 * quat.y * quat.z + 2 * quat.w * quat.x;
    out.m_row[2].z = 1 - 2 * quat2.x - 2 * quat2.y;
    out.m_row[2].w = 0.f;

    return out;
}


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



__device__ bool set_face_normal(float3 dir, float3 outward_normal) {
    bool front_face = getDotProduct(dir, outward_normal) < 0;
    return front_face;
}
__device__ bool aabb(float3 o, float3 d, float3 a, float3 b) {
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
    return true;
}


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
__device__ float3 hit_rect(float t_min, float t_max, float3 origin, float3 dir, double x0, double x1, double y0, double y1, double k) {

    float3 outward_normal = make_float3(0, 0, 1);
    float norm = set_face_normal(dir, outward_normal);
    float t = (k - origin.z) /dir.z;
    if (t < t_min || t > t_max)
        return make_float3(-1,norm,0);
    auto x = origin.x + t *dir.x;
    auto y = origin.y + t *dir.y;
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return make_float3(-1, norm, 0);
   
  
 
  
  
    return make_float3(t,norm,0);
}

__device__ float3 hit_tri(float3 rayOrigin,
    float3 rayVector,
   float3 vertex0, float3 vertex1, float3 vertex2)
{
 
 
        const float EPSILON = 0.01;
        float3 v0v1 = vertex1 - vertex0;
        float3 v0v2 = vertex2 - vertex0;
        // no need to normalize
        float3 N = getCrossProduct(v0v1, v0v2);
        float norm = set_face_normal(rayVector, N);
        float3 edge1, edge2, h, s, q;
        float a, f, u, v;
        edge1 = vertex1 - vertex0;
        edge2 = vertex2 - vertex0;
        h = getCrossProduct(rayVector, edge2);
        a = getDotProduct(edge1, h);
        if (a > -EPSILON && a < EPSILON)
            return make_float3(-1, norm, 0);    // This ray is parallel to this triangle.
        f = 1.0 / a;
        s = rayOrigin - vertex0;
        u = f * getDotProduct(s, h);
        if (u < 0.0 || u > 1.0)
            return make_float3(-1, norm, 0);
        q = getCrossProduct(s, edge1);
        v = f * getDotProduct(rayVector, q);
        if (v < 0.0 || u + v > 1.0)
            return make_float3(-1, norm, 0);
        // At this stage we can compute t to find out where the intersection point is on the line.
        float t = f * getDotProduct(edge2, q);
        if (t > EPSILON) // ray intersection
        {

            return make_float3(t, norm, 0);
        }
        else // This means that there is a line intersection but not a ray intersection.
            return make_float3(-1, norm, 0);

   
}
__device__ float3 hit_sphere(const float3 center, float radius, float3 origin, float3 dir) {
    float3 off = make3(radius+0.1);
    if (aabb(origin, dir, center - off, center + off)) {
        float3 oc = origin - center;
        float a = pow(getLength(dir), 2.0f);
        float half_b = getDotProduct(oc, dir);
        float3 outward_normal = (origin - center) / make3(radius);
        float norm = set_face_normal(dir, outward_normal);
        float c = pow(getLength(oc), 2.0f) - radius * radius;
        float discriminant = half_b * half_b - a * c;
        if (discriminant < 0) {
            return make_float3(-1.0, norm, 0);
        }
        else {
            return make_float3((-half_b - sqrt(discriminant)) / a, norm, 0);
        }
    } else{

        return make_float3(-1.0, 0, 0);
    }
}


__device__ void set3(int w, float3 x, float3 y, float3 z, float3 *outx, float3* outy, float3* outz) {
    outx[w] = x;
    outy[w] = y;
    outz[w] = z;

}



__device__ float3 hit_cube(float3 origin, float3 dir, float3 position, float3 dimension, float3 &vex, float3 rotation, float3& p, float3& newdir) {
  
    float mindist = 10000;
    bool yes = true;
    float closest = 0;
    float normaldir = 0;
    int which = 0;
    p = origin;
    newdir = dir;
    origin = origin - position;
 
    position = make3(0);

   
  

    if (rotation.x > 0) {
        float3 originn = origin;
        float3 dirr = dir;



        float radians = rotation.x * M_PI / 180;

        float sin_theta = sin(radians);
        float cos_theta = cos(radians);


        originn.y = cos_theta * origin.y - sin_theta * origin.z;
        originn.z = sin_theta * origin.y + cos_theta * origin.z;


          origin = originn;
        dirr.y = cos_theta * dir.y - sin_theta * dir.z;
        dirr.z = sin_theta * dir.y + cos_theta * dir.z;

      
        dir = dirr;

    }


    if (rotation.y > 0) {
        float3 originn = origin;
        float3 dirr = dir;



        float radians = rotation.y * M_PI / 180;

        float sin_theta = sin(radians);
        float cos_theta = cos(radians);


        originn.x = cos_theta * origin.x - sin_theta * (origin.z);
        originn.z = sin_theta * origin.x + cos_theta * (origin.z);
        origin = originn;


        dirr.x = cos_theta * dir.x - sin_theta * dir.z;
        dirr.z = sin_theta * dir.x + cos_theta * dir.z;

      
        dir = dirr;

    }
    if (rotation.z > 0) {
        float3 originn = origin;
        float3 dirr = dir;



        float radians = rotation.z * M_PI / 180;

        float sin_theta = sin(radians);
        float cos_theta = cos(radians);


        originn.x = cos_theta * origin.x - sin_theta * (origin.y);
        originn.y = sin_theta * origin.x + cos_theta * (origin.y);

        origin = originn;

        dirr.x = cos_theta * dir.x - sin_theta * dir.y;
        dirr.y = sin_theta * dir.x + cos_theta * dir.y;

       
        dir = dirr;

    }
   
 
   
      
            float3 dists[12];
            float3 x[12];
            float3 y[12];
            float3 z[12];
         
            
            dists[0] = hit_tri(origin, dir, position + (make_float3( -1,-1,1)*dimension), position + (make_float3(1, -1, 1) * dimension), position + (make_float3(1, 1, 1) * dimension));
            set3(0, position + (make_float3(-1, -1, 1) * dimension), position + (make_float3(1, -1, 1) * dimension), position + (make_float3(1, 1, 1) * dimension), x, y, z);
            
            dists[1] = hit_tri(origin, dir, position + (make_float3(-1, -1, 1) * dimension), position + (make_float3(1, 1, 1) * dimension), position + (make_float3(-1, 1, 1) * dimension));
            set3(1, position + (make_float3(-1, -1, 1) * dimension), position + (make_float3(1, 1, 1) * dimension), position + (make_float3(-1, 1, 1) * dimension), x, y, z);
            dists[2] = hit_tri(origin, dir, position + (make_float3(1, -1, 1) * dimension), position + (make_float3(1, -1, -1) * dimension), position + (make_float3(1, 1, -1) * dimension));
            set3(2, position + (make_float3(1, -1, 1) * dimension), position + (make_float3(1, -1, -1) * dimension), position + (make_float3(1, 1, -1) * dimension), x, y, z);

            dists[3] = hit_tri(origin, dir, position + (make_float3(1, -1, 1) * dimension), position + (make_float3(1, 1, -1) * dimension), position + (make_float3(1, 1, 1) * dimension));
            set3(3, position + (make_float3(1, -1, 1) * dimension), position + (make_float3(1, 1, -1) * dimension), position + (make_float3(1, 1, 1) * dimension), x, y, z);

            dists[4] = hit_tri(origin, dir, position + (make_float3(1, -1, -1) * dimension), position + (make_float3(-1, -1, -1) * dimension), position + (make_float3(-1, 1, -1) * dimension));

            set3(4, position + (make_float3(1, -1, -1) * dimension), position + (make_float3(-1, -1, -1) * dimension), position + (make_float3(-1, 1, -1) * dimension), x, y, z);
            dists[5] = hit_tri(origin, dir, position + (make_float3(1, -1, -1) * dimension), position + (make_float3(-1, 1, -1) * dimension), position + (make_float3(1, 1, -1) * dimension));
            set3(5, position + (make_float3(1, -1, -1) * dimension), position + (make_float3(-1, 1, -1) * dimension), position + (make_float3(1, 1, -1) * dimension), x, y, z);

            dists[6] = hit_tri(origin, dir, position + (make_float3(-1, -1, -1) * dimension), position + (make_float3(-1, -1, 1) * dimension), position + (make_float3(-1, 1, 1) * dimension));
            set3(6, position + (make_float3(-1, -1, -1) * dimension), position + (make_float3(-1, -1, 1) * dimension), position + (make_float3(-1, 1, 1) * dimension), x, y, z);

            dists[7] = hit_tri(origin, dir, position + (make_float3(-1, -1, -1) * dimension), position + (make_float3(-1, 1, 1) * dimension), position + (make_float3(-1, 1, -1) * dimension));
            set3(7, position + (make_float3(-1, -1, -1) * dimension), position + (make_float3(-1, 1, 1) * dimension), position + (make_float3(-1, 1, -1) * dimension), x, y, z);
            dists[8] = hit_tri(origin, dir, position + (make_float3(-1, 1, 1) * dimension), position + (make_float3(1, 1, 1) * dimension), position + (make_float3(1, 1, -1) * dimension));
            set3(8, position + (make_float3(-1, 1, 1) * dimension), position + (make_float3(1, 1, 1) * dimension), position + (make_float3(1, 1, -1) * dimension), x, y, z);
            dists[9] = hit_tri(origin, dir, position + (make_float3(-1, 1, 1) * dimension), position + (make_float3(1, 1, -1) * dimension), position + (make_float3(-1, 1, -1) * dimension));
            set3(9, position + (make_float3(-1, 1, 1) * dimension), position + (make_float3(1, 1, -1) * dimension), position + (make_float3(-1, 1, -1) * dimension), x, y, z);
            dists[10] = hit_tri(origin, dir, position + (make_float3(1, -1, 1) * dimension), position + (make_float3(-1, -1, -1) * dimension), position + (make_float3(1, -1, -1) * dimension));
            set3(10, position + (make_float3(1, -1, 1) * dimension), position + (make_float3(-1, -1, -1) * dimension), position + (make_float3(1, -1, -1) * dimension), x, y, z);
         dists[11] = hit_tri(origin, dir, position + (make_float3(1, -1, 1) * dimension), position + (make_float3(-1, -1, 1) * dimension), position + (make_float3(-1, -1, -1) * dimension));
         set3(11, position + (make_float3(1, -1, 1) * dimension), position + (make_float3(-1, -1, 1) * dimension), position + (make_float3(-1, -1, -1) * dimension), x, y, z);
            for (int xx = 0; xx < 12; xx++) {
                if (dists[xx].x < mindist && dists[xx].x > -0.05) {
                    mindist = dists[xx].x;
                    which = xx;
                    normaldir = dists[xx].y;
                   
                    yes = false;
                   
                }

            }

            float3 vertex0 = x[which];
            float3 vertex1 = y[which];
            float3 vertex2 = z[which];


            float3 v0v1 = vertex1 - vertex0;
            float3 v0v2 = vertex2 - vertex0;
            // no need to normalize
            float3 N = getNormalizedVec(getCrossProduct(v0v1, v0v2));

          //  p = origin + (make3(mindist) * dir);
         //   p = p + position;
            if (rotation.x > 0) {
                float3 nn = N;
                float radians = rotation.x * M_PI / 180;

                float sin_theta = sin(radians);
                float cos_theta = cos(radians);

                nn.y = cos_theta * N.y + sin_theta * N.z;
                nn.z = -sin_theta * N.y + cos_theta * N.z;
                N = nn;


                


            }
            if (rotation.y > 0) {
                float3 nn = N;
                float radians = rotation.y * M_PI / 180;

                float sin_theta = sin(radians);
                float cos_theta = cos(radians);

                nn.x = cos_theta * N.x + sin_theta * N.z;
                nn.z = -sin_theta * N.x + cos_theta * N.z;
                N = nn;

              
            }
            if (rotation.z > 0) {
                float3 nn = N;
                float radians = rotation.z * M_PI / 180;

                float sin_theta = sin(radians);
                float cos_theta = cos(radians);

                nn.x = cos_theta * N.x + sin_theta * N.y;
                nn.y = -sin_theta * N.x + cos_theta * N.y;
                N = nn;

             
            }
            vex = N;
          //normaldir = set_face_normal(dir, N);
            if (yes) {

                mindist = -1;
            }
      
            return make_float3(mindist, normaldir, which);
            

    
   
    
  return make_float3(-1.0, 0, 0);


}


bool bounding_box(int obj, float3& min, float3& max, singleobject* b) {



    if (b[obj].type == 0) {

        min = b[obj].pos - make3(b[obj].dim.x);
        max = b[obj].pos + make3(b[obj].dim.x);
    }

    else  if (b[obj].type == 1) {
        min = make_float3(b[obj].pos.x, b[obj].pos.y, b[obj].pos.z - 0.0001);
        max = make_float3(b[obj].pos.x + b[obj].dim.x, b[obj].pos.y + b[obj].dim.y, b[obj].pos.z + 0.0001);



    }

    else  if (b[obj].type == 2) {
        float3 v1 = b[obj].pos;
        float3 v2 = b[obj].dim;
        float3 v3 = b[obj].rot;
        float minx = fmin(v1.x, fmin(v2.x, v3.x));

        float miny = fmin(v1.y, fmin(v2.y, v3.y));
        float minz = fmin(v1.z, fmin(v2.z, v3.z));
        float maxx = fmax(v1.x, fmax(v2.x, v3.x));
        float  maxy = fmax(v1.y, fmax(v2.y, v3.y));
        float  maxz = fmax(v1.z, fmax(v2.z, v3.z));


        min = make_float3(minx - 0.01, miny - 0.01, minz - 0.01);
        max = make_float3(maxx + 0.01, maxy + 0.01, maxz + 0.01);


    }


    else  if (b[obj].type == 3) {

        min = b[obj].pos - make3(getLength(b[obj].dim));
        max = b[obj].pos + make3(getLength(b[obj].dim));

    }

    return true;

}


__device__ bool dbounding_box(int obj, float3& min, float3& max, singleobject* b) {



    if (b[obj].type == 0) {

        min = b[obj].pos - make3(b[obj].dim.x);
        max = b[obj].pos + make3(b[obj].dim.x);
    }

    else  if (b[obj].type == 1) {
        min = make_float3(b[obj].pos.x, b[obj].pos.y, b[obj].pos.z - 0.0001);
        max = make_float3(b[obj].pos.x + b[obj].dim.x, b[obj].pos.y + b[obj].dim.y, b[obj].pos.z + 0.0001);



    }

    else  if (b[obj].type == 2) {
        float3 v1 = b[obj].pos;
        float3 v2 = b[obj].dim;
        float3 v3 = b[obj].rot;
        float minx = fmin(v1.x, fmin(v2.x, v3.x));

        float miny = fmin(v1.y, fmin(v2.y, v3.y));
        float minz = fmin(v1.z, fmin(v2.z, v3.z));
        float maxx = fmax(v1.x, fmax(v2.x, v3.x));
        float  maxy = fmax(v1.y, fmax(v2.y, v3.y));
        float  maxz = fmax(v1.z, fmax(v2.z, v3.z));


        min = make_float3(minx - 0.01, miny - 0.01, minz - 0.01);
        max = make_float3(maxx + 0.01, maxy + 0.01, maxz + 0.01);


    }


    else  if (b[obj].type == 3) {

        min = b[obj].pos - make3(getLength(b[obj].dim));
        max = b[obj].pos + make3(getLength(b[obj].dim));

    }

    return true;

}


 void surrounding_box(float3 amin, float3 amax, float3 bmin, float3 bmax, float3 &min,float3 &max) {
    min = make_float3(fmin(amin.x, bmin.x),
        fmin(amin.y, bmin.y),
        fmin(amin.z, bmin.z));

   max = make_float3(fmax(amax.x, bmax.x),
        fmax(amax.y, bmax.y),
        fmax(amax.z, bmax.z));

   
}
 bool arraybound(float3 &min, float3 &max, int objs[], int len, singleobject* b) {
    if (len==0) return false;

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

 __device__ float3 bvhhit(float3 origin, float3 dir, float3& vex, float3& rotpoint, float3& newdir, bvh* bvhtree) {




    if (edebugnum[0] < 0 || edebugnum[0] >= dbvhnumnum[0]) {

        return make_float3(-1, 0, 0);
    }



    bvh currentnode = bvhtree[abs(edebugnum[0])];



    if (aabb(origin, dir, currentnode.min, currentnode.max)) {


        return  make_float3(1, 0, 0);



    }




    return make_float3(-1, 0, 0);

}








__device__ float3 singlehit(float3 origin, float3 dir, float3& vex, float3& rotpoint, float3& newdir, int x, singleobject* b) {


    float mindist = 10000;
    bool yes = true;
    float closest = 0;
    float normaldir = 0;

    float3 dist;
    float3 bon;
    if (b[x].type == 0) {
        dist = hit_sphere(b[x].pos, b[x].dim.x, origin, dir);

    }
    else  if (b[x].type == 1) {
        dist = hit_rect(0, 1000, origin, dir, b[x].pos.x, b[x].pos.x + b[x].dim.x, b[x].pos.y, b[x].pos.y + b[x].dim.y, b[x].pos.z);

    }
    else  if (b[x].type == 2) {
        dist = hit_tri(origin, dir, b[x].pos, b[x].dim, b[x].rot);

    }
    else  if (b[x].type == 3) {

        dist = hit_cube(origin, dir, b[x].pos, b[x].dim, bon, b[x].rot, rotpoint, newdir);

    }

    if (dist.x < mindist && dist.x > -0.0) {
        vex = bon;
        mindist = dist.x;
        closest = x;
        normaldir = dist.y;
        yes = false;
    }

    if (yes) {

        mindist = -1;
    }
    return make_float3(mindist, closest, normaldir);


}



__device__ float3 hit(float3 origin, float3 dir, float3& vex, float3& rotpoint, float3& newdir, bvh* bvhtree, singleobject* b) {


    if (bvhhit(origin, dir, vex, rotpoint, newdir, bvhtree).x > -0.1) {


        return  make_float3(1, 2, 1);;

    }





    int tracked[10000];
    tracked[0] = 0;
    int num = 1;
    int mini = 0;
    float3 out = make_float3(10000000, 0, 0);
    bool oof = true;
    int i = 0;
    float closest = 100000000;

    while (mini < num) {
        i++;
        if (i > anum[0]) {
            break;
        }


        //get array length
        int numm = num;
        for (int node = mini; node < numm; node++) {








            //! this seems to have a huge performsance impact! Seems to be the problem with fps











            //remove from array
            mini++;

            float dister;
            if (aabb2(origin, dir, bvhtree[tracked[node]].min, bvhtree[tracked[node]].max,dister)) {
                if (dister < closest) {
                   
                    if (bvhtree[tracked[node]].end == true) {

                        float3 temp = singlehit(origin, dir, vex, rotpoint, newdir, bvhtree[tracked[node]].under, b);
                        if (temp.x > -0.01) {
                            if (temp.x < out.x) {
                                out = temp;
                                oof = false;
                                closest = out.x;
                            }


                        }

                    }
                    else {

                        tracked[num] = bvhtree[tracked[node]].children[0];
                        num++;
                        tracked[num] = bvhtree[tracked[node]].children[1];
                        num++;
                    }



                }
              

                //add to array






            }



        }

    }
    if (oof == true) {
        out = make_float3(-1, 0, 0);

    }

    return out;

}



















__device__ float3 random_in_unit_sphere() {
    while (true) {
        curandState state;
        int tId = threadIdx.x + (blockIdx.x * blockDim.x);

        curand_init((unsigned long long)clock() + tId, 0, 0, &state);

        double rand1 = (curand_uniform_double(&state)*2)-1;
        double rand2 = (curand_uniform_double(&state)*2)-1;
        double rand3 = (curand_uniform_double(&state)*2)-1;
        auto p = make_float3(rand1,rand2,rand3);
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
__device__ float3 reflect( float3 v, float3 n) {
    return v - make_float3(2.0 * getDotProduct(v, n), 2.0 * getDotProduct(v, n), 2.0 * getDotProduct(v, n)) * n;
}
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
__device__ float3 getnormal(int obj, float3 origin, float3 hitpoint, float3 vex, singleobject* b, float3 dir, float3 &texco) {

    

    if (b[obj].type == 0) {
        return (hitpoint - b[obj].pos) / make3(b[obj].dim.x);

    }

 else  if (b[obj].type == 1) {
 
        return make_float3(0, 0, 1);
        }

 else  if (b[obj].type == 2) {
        float3 uv;

        float3 vertex0 = b[obj].pos;
        float3 vertex1 = b[obj].dim;
        float3 vertex2 = b[obj].rot;
      

        float3 v0v1 = vertex1 - vertex0;
        float3 v0v2 = vertex2 - vertex0;
        // no need to normalize
        float3 N =getCrossProduct(v0v1, v0v2);

        /*
        float d0 = distance(vertex0, hitpoint);
        float d1 = distance(vertex1, hitpoint);
        float d2 = distance(vertex2, hitpoint);

        // Our three points.. re-oriented so that 'a' is the farthest

        float3 a, bb, c;
        float3 na, nb, nc;
        if (d0 > d1 && d0 > d2)
        {
            a = vertex0;
            bb = vertex1;
            c = vertex2;
            na = n0;
            nb = n1;
            nc = n2;
        }
        else if (d1 > d0 && d1 > d2)
        {
            a = vertex1;
            bb = vertex0;
            c = vertex2;
            na = n1;
            nb = n0;
            nc = n2;
        }
        else // if (d2 > d0 && d2 > d1)
        {
            a = vertex2;
            bb = vertex0;
            c = vertex1;
            na = n2;
            nb = n0;
            nc = n1;
        }
        // Our three points.. re-oriented so that 'a' is the farthest
       vertex0= a;
        vertex1 = bb;
       vertex2 = c;
       n0 = na;
       n1= nb;
         n2 = nc;


        v0v1 = vertex1 - vertex0;
        v0v2 = vertex2 - vertex0;

     */
      
        float3 pvec = getCrossProduct(dir,v0v2);
        float det = getDotProduct(v0v1,pvec);

        // ray and triangle are parallel if det is close to 0
       

        float invDet = 1 / det;

        float3 tvec = origin - vertex0;
        uv.x = getDotProduct(tvec,pvec) * invDet;
     

        float3 qvec =getCrossProduct(tvec,v0v1);
        uv.y = getDotProduct(dir,qvec) * invDet;
      

        uv.z = 1 - uv.x - uv.y;

        texco = make3(uv.z) * b[obj].t1 + make3(uv.x) * b[obj].t2 + make3(uv.y ) * b[obj].t3;
        
        if (b[obj].norm.z != -20) {
            N = b[obj].norm;

            
            
           
            if (b[obj].n1.z != -20 && b[obj].smooth) {
               
                float3 n0 = b[obj].n1;
                float3 n1 = b[obj].n2;
                float3 n2 = b[obj].n3;
                N = make3(uv.z) * n0 + make3(uv.x) * n1 + make3(uv.y) * n2;
            }
           

        }
        return  getNormalizedVec(N);
    }

   
 else  if (b[obj].type == 3) {
   
        return vex;

        }
    return getNormalizedVec(hitpoint - b[obj].pos);


}


__device__ float3 checker(float3 uv, float3 p, float3 col1, float3 col2) {
    float u2 = floor(uv.x * 10);
    float v2 = floor(uv.y * 10);
    float yes = u2 + v2;
    if (fmod(yes,(float)2) == 0)
        return col1;
    else
        return col2;
}
__device__ float3 raycolor(float3 origin,float3 dir, int max_depth, bvh* bvhtree, singleobject* b, cudaTextureObject_t* tex) {
    float3 raydir = dir;
    float3 rayo = origin;
    float3 cur_attenuation = make3(1.0f);
    
    for (int i = 0; i < max_depth; i++) {
        float3 rotpoint;
        float3 newdir = raydir;
        float3 vex;

        float3 texco;

        float3 hitoride = hit(rayo, raydir, vex, rotpoint, newdir,bvhtree, b);

        int g = int(hitoride.y);
        float hit = hitoride.x;
        if (hit > 0.0) {
         
            int matter = b[g].mat;
            float3 centerofobject = b[g].pos;
            float3 hitt = make3(hit);



      
            if (b[g].type == 3) {

                rayo = rotpoint;
                raydir = newdir;
            }
            
            float3 hitpoint = rayo + (hitt * raydir);
          
          
            float3 N =  getnormal(g, rayo, hitpoint,vex, b, raydir, texco);
            bool inorout = set_face_normal(dir, N);
          //  float3 N = getNormalizedVec(hitpoint - centerofobject);
            N = inorout ? N : N * make3(-1);

            float3 ocolor = b[g].col;

            if (b[g].tex) {

               ocolor =  checker(texco, hitpoint, make3(0.8), b[g].col);
           
            
           
              
            }
            if (b[g].texnum >= 0) {
                uchar4 C = tex2D<uchar4>(tex[b[g].texnum], texco.x, -texco.y+1);
               // N = getNormalizedVec(N * make_float3(float(C.x) , float(C.y) , float(C.z) ));
                ocolor = make_float3(float(C.x) / 255, float(C.y) / 255, float(C.z) / 255);
            }
        
            if (matter == 0) {
                float3 target = hitpoint + N;
                    if (b[g].addional.x == 0) {
                        target = target+ random_in_unit_sphere();
                    }
                    else {
                        target = target + getNormalizedVec( random_in_unit_sphere());

                    }

               

                   
                // checker pattern
               

                    cur_attenuation = cur_attenuation  * ocolor;
              
               
                   

                
              
                
                rayo = hitpoint;
                raydir = getNormalizedVec(target - hitpoint);
               
            }
            else if (matter == 2) {
                float3 reflected = reflect(getNormalizedVec(raydir), N);
                cur_attenuation = cur_attenuation * ocolor;
                rayo = hitpoint;
                raydir = reflected;

            }
            else if (matter == 3) {
                float3 reflected = reflect(getNormalizedVec(raydir), N);
                cur_attenuation = cur_attenuation * ocolor;
                rayo = hitpoint;
                
                raydir = reflected + make3(b[g].addional.y) * random_in_unit_sphere();

            }
          
            else if (matter == 4) {
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
                /*
                if (inorout == 0) {

                    return make3(1) * cur_attenuation;
                }
                */
                return ocolor * cur_attenuation;
            }
     
             //return make_float3(N.x , N.y , N.z );



        }
        else {
            

            float3 unit_direction = getNormalizedVec(raydir);
            float t = 0.5 * (unit_direction.y + 1.0);
            float3 c = make_float3((1.0 - t), (1.0 - t), (1.0 - t)) * make_float3(1.0, 1.0, 1.0) + make_float3(t, t, t) * make_float3(0.5, 0.7, 1.0);
            return cur_attenuation * c * make_float3(backgroundintensity[0], backgroundintensity[0], backgroundintensity[0]);

        }
       
        
    }
    return make_float3(0.0, 0.0, 0.0);
}
#endif  // UNIFIED_MATH_CUDA_H

#include <math.h>

//utiltiy functions

/*
__device__ bool hit_sphere(float3 center, float radius, float3 origin, float3 dir) {
    float b = 2 * getDotProduct(dir, make_float3(origin.x - center.x, origin.y - center.y, origin.z - center.z));
    float3 norm =make_float3(origin.x - center.x, origin.y - center.y, origin.z - center.z);
    float yes = sqrt(norm.x * norm.x + norm.y * norm.y + norm.z * norm.z);

    float c = pow(yes, 2.0f) - pow(radius, 2.);
    float delta = pow(b, 2) - 4 * c;
    if (delta > 0) {
        float t1 = (-b + sqrt(delta)) / 2;
        float t2 = (-b - sqrt(delta)) / 2;
        if (t1 > 0 && t2 > 0) {

            return true;
        }


    }

    return false;
}*/

//make sure to chnage retrunt from float whatever[3] to float* whatever = new float[3] and thendelete it later where it is returned

//ray function. Called for every pixel. Kind of like shader

__device__ float3 random_in_unit_disk() {
    while (true) {
        float3 p = make_float3((randy()*2)-1, (randy() * 2) - 1, 0);
        if (pow(getLength(p), 2.0f) >= 1) continue;
        return p;
    }
}
__global__ void Kernel(int* outputr, int* outputg, int* outputb, float* settings, bvh* bvhtree, singleobject* b, cudaTextureObject_t* tex)
{
   
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = x * SCREEN_HEIGHT + y;
    float aspect = float(SCREEN_WIDTH / settings[11]) / float(SCREEN_HEIGHT / settings[11]);
    float u =( float(x) / float(SCREEN_WIDTH / settings[11]));
    float v =float(y) / float(SCREEN_HEIGHT / settings[11]);
    // * aspect
    if ((x >= SCREEN_WIDTH / settings[11]) || (y >= SCREEN_HEIGHT / settings[11])) return;
    outputr[w] = 0;
    outputg[w] = 0;
    outputb[w] = 0;
    float fov = settings[8];
    float fovvv = fov* M_PI/180;
   
    float h = tan(fovvv / 2);
    float viewport_height = 2.0 * h;
    float viewport_width = aspect * viewport_height;
 
    /* float3 cam = make_float3(0, 0, 2);

    float3 origin = cam;
    float3 horizontal = make_float3(SCREEN_WIDTH, 0, 0);
    float3 vertical = make_float3(0, SCREEN_HEIGHT, 0);
    float3 lower_left_corner = origin - horizontal / make_float3(2.0f, 2.0f, 2.0f) - vertical / make_float3(2.0f, 2.0f, 2.0f) - make_float3(0.0f, 0.0f, focal_length);*/
    
    //float3 dir = getNormalizedVec(pixel-origin);
    float3 lookfrom = make_float3(settings[0], settings[1], settings[2]);
    float3 lookat = make_float3(settings[3], settings[4], settings[5]);
    float3 vup = make_float3(0, 1, 0);
 
    float aperture = settings[6];
    float focus_dist = settings[7];

    float3 wu = getNormalizedVec(lookfrom - lookat) ;
    float3 uu = getNormalizedVec(getCrossProduct(vup, wu));
    float3 vu = getCrossProduct(wu, uu);

    float3 origin = lookfrom;
    float3 horizontal = make3(focus_dist) * make3(viewport_width) * uu;
    float3 vertical = make3(focus_dist) * make3(viewport_height) * vu;
    float3 lower_left_corner = origin - horizontal / make3(2) - vertical / make3(2) - make3(focus_dist) * wu ;

        
    float lens_radius = aperture / 2;


    int samples_per_pixel = settings[10];
    int max_depth = settings[9];

    float3 N = make_float3(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; ++s) {
        int tId = threadIdx.x + (blockIdx.x * blockDim.x);
        curandState state;
        curand_init((unsigned long long)clock() + tId, 0, 0, &state);

        double rand1 = curand_uniform_double(&state);
        double rand2 = curand_uniform_double(&state);
        float nu = ((float(x) + rand1) / float(SCREEN_WIDTH / settings[11]));
        float nv = (float(y) + rand2) / float(SCREEN_HEIGHT / settings[11]);
        float3 rd = make3(lens_radius) * random_in_unit_disk();
        float3 offset = uu * make3(rd.x) + vu * make3(rd.y);
        float3 dir = lower_left_corner + make3(nu) * horizontal + make3(nv) * vertical - origin-offset;
        N = N + raycolor(origin+offset, dir, max_depth,  bvhtree, b,tex);

    }
    float scale = 1.0 / samples_per_pixel;
        outputr[w] = N.x * 255*scale;
        outputg[w] = N.y * 255 * scale;
        outputb[w] = N.z * 255 * scale;


    
    // Add vectors in parallel.



}




 double random_double() {
    // Returns a random real in [0,1).
     srand(GetTickCount());
    return rand() / (RAND_MAX + 1.0);
}

 double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_double();
}








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
             // Output the text from the file
            
            //create string stream from the string
             if (myText[0] == "/"[0])
                 continue;
             if (myText[0] == "*"[0]) {
               
                 continue;


             }
           
             line++;
         }
       
      
         // Close the file
         MyReadFile.close();
         return line + 1;
     }
     else {

         SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR,
             "Missing file",
             "scene.txt is missing.",
             NULL);
         return 0;
     }

 }


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
            // Output the text from the file
            int colum = 0;
            stringstream s_stream(myText); //create string stream from the string
            if(myText[0] == "/"[0])
            continue;
            if (myText[0] == "*"[0]) {
                while (s_stream.good()) {
                    string substr;
                    getline(s_stream, substr, ','); //get first string delimited by comma

                      //  *,camx,camy,camz,apeture, lookx,looky,lookz,focus dist, fov, max depth, samples per frame
                    if (colum == 0) {
                       
                    }
                    else if (colum == 1) {

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

                        fovv = stof(substr);
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
                  


                    colum++;
                }
                continue;


            }
            while (s_stream.good()) {
                string substr;
                getline(s_stream, substr, ','); //get first string delimited by comma
           
                  
                if (substr == "r") {
                 
                    double r =random_double();
                    substr = to_string(r);
                  

                }
                  // x,y,z,type, r,g,b,extra, lam, dimx,dimy,dimz,mat
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
        nanum[0] = line+1;
        // Close the file
        MyReadFile.close();
    }
    else {

        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR,
            "Missing file",
            "scene.txt is missing.",
            NULL);
    }

}
 int random_int(int min, int max) {
    // Returns a random integer in [min,max].
     srand(GetTickCount());
    return static_cast<int>(random_double(min, max + 1));
}

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
void sorto(float* output, float3 input[], int size, int* yett) {
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



   //int axis =  random_int(0, 2);


  

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
   
    pairsort(output, yett, size);

    /*
    float* first(&output[0]);
    float* last(first + size);
    std::sort(first, last);
    */
}

void split(int input[],int* a, int* b, int num, singleobject* bb) {

    float3 *many = new float3[num];
    int* aoutput = new int[num];
    for (int o = 0; o < num; ++o) {

        many[o] = bb[input[o]].pos;
        aoutput[o] = input[o];
    }

   


    float* output = new float[num];
    sorto(output, many, num, aoutput);
    
   


    int part1 = num / 2;
 
    for (int o = 0; o < part1; ++o) {

        a[o] = aoutput[o];

    }
    for (int o = part1; o < num; ++o) {

        b[o-part1] = aoutput[o];

    }


    delete[] output;
    delete[] aoutput;
    delete[] many;

}


void build_bvh(bvh* nbvhtree, singleobject* bb) {

    //build bounding volume heirarchy
    bvhunder* under = new bvhunder[bvhnum];

    //clear bvh
    bvh defualt;
    defualt.active = false;
 
    std::fill_n(nbvhtree, bvhnum, defualt);
    actualbvhnum = 1;

    int iteration = 0;


    //set up first node (top down)
    //asighn all avluies exept children(we will set when we proccess childeren)

    for (int o = 0; o < nanum[0]; ++o) {



        under[0].under[o] = o;

    }
    nbvhtree[0].active = true;
    nbvhtree[0].id = 0;
    nbvhtree[0].count = nanum[0]-1;
    nbvhtree[0].end = false;
    float3 firstmin;
    float3 firstmax;
    arraybound(firstmin, firstmax, under[0].under, nanum[0],bb);
   
    nbvhtree[0].min = firstmin;
    nbvhtree[0].max = firstmax;
   
  
    int sorted = nanum[0];
    //for each active node with id==iteration
   //calculate split size
  //split into two arrays
   //create two bvh nodes 
                 //if bvh node has only one object in array mark as end    minus 1 from sorted 
                //calculate bounding boxes
                 //assign enw bvh nodes as the children of the main bvh node
                   //add to actual bvh num

    for (iteration; iteration < 100000; ++iteration) {

        if (sorted <= 0) {

            break;
        }
        int boi = actualbvhnum;
        for (int node = 0; node < boi; ++node) {

            if (nbvhtree[node].active == true && nbvhtree[node].id == iteration && nbvhtree[node].end == false) {

                int size = nbvhtree[node].count;
                int partition1 = size / 2;
                int partition2 = size - partition1;
             //   std::cout << size << "\n";

                int *a = new int[partition1];
                int *b = new int[partition2];

                split(under[node].under, a,  b, size, bb);
                
                //create node 1    yes
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
                 //   std::cout << "end reached: " << node1.under[0] << "\n";
                }


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
                   // std::cout << "end reached: " << node2.under[0] << "\n";
                }
             

                float3 bmin;
                float3 bmax;
                arraybound(bmin, bmax, b, partition2, bb);
                node2.min = bmin;
                node2.max = bmax;
                nbvhtree[node].children[1] = actualbvhnum-1;
                nbvhtree[actualbvhnum-1] = node2;
         

               
                

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
    std::cout << objnum << " objects/faces total" << std::endl;



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
    SDL_Surface* screenSurface = NULL;
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
                        //handle exi throug escape
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
    float settings[12] = { campos.x , campos.y,campos.z, look.x,  look.y,  look.z, aperturee ,focus_diste,fovv, max_depthh, samples_per_pixell,divisor };

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

    cudaStatus = cudaMalloc((void**)&dev_settings, 12 * sizeof(float));

    cudaStatus = cudaMalloc((void**)&dev_bvhtree, bvhnum * sizeof(bvh));
    cudaStatus = cudaMalloc((void**)&dev_allobjects, objnum * sizeof(singleobject));
    cudaStatus = cudaMalloc((void**)&dev_texarray, texnum * sizeof(cudaTextureObject_t));

    //get device
    int device = -1;
    cudaGetDevice(&device);

    // Copy input vectors from host memory to GPU buffers(and prefetch?)
    cudaStatus = cudaMemcpy(dev_settings, settings, 12 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemPrefetchAsync(dev_settings, 12 * sizeof(float), device, NULL);

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

