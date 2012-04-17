#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <time.h>
const int width = 2048;
const int height = width;
const int maxReflect = 5;
struct Color {
  unsigned char b, g, r, a;
};

Color* image;
void writebmpheader(FILE* f, int width, int height) {
  int size = width * height * sizeof(Color);
  struct {
    uint32_t filesz;
    uint16_t creator1;
    uint16_t creator2;
    uint32_t bmp_offset;
  } bmpheader = 
    { size + 54, 0, 0, 54};
  struct {
    uint32_t header_sz;
    int32_t width;
    int32_t height;
    uint16_t nplanes;
    uint16_t bitspp;
    uint32_t compress_type;
    uint32_t bmp_bytesz;
    int32_t hres;
    int32_t vres;
    uint32_t ncolors;
    uint32_t nimpcolors;    
  } dibheader = 
    {40, width, height, 1, 32, 0, size, 0, 0, 0, 0};
  fwrite("BM", 2, 1, f);
  fwrite(&bmpheader, sizeof(bmpheader), 1, f);
  fwrite(&dibheader, sizeof(dibheader), 1, f);
}
void writebmp(const char* filename, const Color* data, int width, int
              height) {
  FILE* f = fopen(filename, "wb");
  if (!f) return;
  writebmpheader(f, width, height);
  fwrite(data, sizeof(Color), width * height, f);
  fclose(f);
}

__device__ __host__ inline float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ inline float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3& operator+=(float3& a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}
__device__ __host__ inline float3 cross(float3 a, float3 b) {
  return make_float3( -a.z * b.y + a.y * b.z,
                      a.z * b.x - a.x * b.z,
                      -a.y * b.x + a.x * b.y );
}
__device__ inline float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __host__ float3 inline operator*(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ __host__ float3 inline operator*(float b, float3 a) {
  return a * b;
}

__device__ __host__ inline float sqrlength(float3 v) {
  return v.x * v.x + v.y * v.y + v.z * v.z;
}
__device__ __host__ inline float length(float3 v) {
  return sqrtf(sqrlength(v));
}
__device__ __host__ inline float3 normalize(float3 v) {
  float invlen = 1 / length(v);
  return make_float3(v.x * invlen, v.y * invlen, v.z * invlen);
}
__device__ inline float3 modulate(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
struct Ray {
  float3 origin, direction;

  __device__ float3 getPoint(float t) {
    return origin + t * direction;
  }
};

struct PerspectiveCamera {
  float3 eye, front, right, up;
  float fovScale;

  __device__ Ray generateRay(float x, float y) const {
    float3 r = right* ((x - 0.5) * fovScale);
    float3 u = up * ((y - 0.5) * fovScale);
    Ray ray = {eye, normalize(front + r + u)};
    return ray;
  }
};

PerspectiveCamera makePerspectiveCamera(float3 e, float3 f, float3 u, float v) {
  PerspectiveCamera c = {e, f, cross(f, u), cross(cross(f,u), f) , tan(v * 0.5 * 3.1415926 / 180) * 2};
  return c;
};
enum g_t {G_SPHERE, G_PLANE} ;
struct IntersectResult {
  g_t g_type;
  int g_id;
  float distance, reflectiveness;
  float3 position, normal;
};

struct Sphere {
  float3 center;
  float radius, sqrRadius;
  float3 diffuse, specular;
  int shininess;
  float reflectiveness;
  float3 lightDir;
  float3 lightColor;

  __device__ inline bool intersect(Ray& ray, IntersectResult& result) const {
    float3 v = ray.origin - center;
    float a0 = sqrlength(v) - sqrRadius;
    float DdotV = dot(ray.direction, v);
    if (DdotV <= 0) {
      float discr = DdotV * DdotV - a0;
      if (discr >= 0) {
        result.g_type = G_SPHERE;
        result.distance = -DdotV - sqrt(discr);
        result.position = ray.getPoint(result.distance);
        result.normal = normalize(result.position - center);
        result.reflectiveness = reflectiveness;
        return true;
      }
    }
    return false;
  }

  __device__ inline float3 sample(Ray ray, float3 position, float3 normal) const {
    
    float NdotL = dot(normal, lightDir);
    float3 H = normalize(lightDir - ray.direction);
    float NdotH = dot(normal, H);
    float3 diffuseTerm = diffuse * fmaxf(NdotL, 0.0);
    float3 specularTerm = specular * __powf(fmaxf(NdotH, 0.0), shininess);
    return modulate(lightColor, diffuseTerm + specularTerm);
    
  }
};

Sphere makeSphere(float3 c, float r, float3 d, float3 sp, int sh, float re = 0.0) {
  Sphere s = {
    c, r, r * r, d, sp, sh ,re,
    normalize(make_float3(1, 1, 1)),
    make_float3(1, 1, 1)
  };
  return s;
};
struct Plane {
  float3 normal, position;
  float scale, reflectiveness;
  __device__ inline bool intersect(Ray ray, IntersectResult& result) const {
    float a = dot(ray.direction, normal);
    if (a >= 0.0)
      return false;
    float b = dot(normal, ray.origin - position);
    float d = -b / a;
    result.g_type = G_PLANE;
    result.distance = d;
    result.position = ray.getPoint(d);
    result.normal = normal;
    result.reflectiveness = reflectiveness;
    return true;
  }

  __device__ inline float3 sample(Ray ray, float3 position, float3 normal) const {
    if (fmodf(fabsf(floorf(position.x * 0.1) + floorf(position.z * scale)), 2) < 1)
      return make_float3(0, 0, 0);
    else
      return make_float3(1, 1, 1);
  }
};

Plane makePlane(float3 n, float d, float s, float r) {
  Plane p = {n, n * d, s, r};
  return p;
};
struct RayTracingParam {
  PerspectiveCamera camera;
  int spheres_n;
  Sphere spheres[10];
  int planes_n;
  Plane planes[10];
  int maxReflect;
} cpuparam = 
{
  makePerspectiveCamera(make_float3(0, 5, 15), make_float3(0, 0, -1),
                    make_float3(0, 1, 0), 90),
  2,
  {makeSphere(make_float3(-10, 10, -10), 10, 
          make_float3(1, 0, 0), make_float3(1, 1, 1), 16, 0.25),
   makeSphere(make_float3(10, 10, -10), 10, 
          make_float3(0, 0, 1), make_float3(1, 1, 1), 16, 0.25)},
   1,
   {makePlane(make_float3(0, 1, 0), 0, 0.1, 0.25)}
};

__constant__ RayTracingParam param;
template <typename T>
__device__ inline bool intersect(T* geometries, int n, Ray r, IntersectResult& result) {
  IntersectResult ir;
  bool ok = false;
  for (int i = 0; i < n; ++i) {
    ir.g_id = i;
    if (geometries[i].intersect(r, ir) && ir.distance < result.distance) {
      result.distance = ir.distance;
      result = ir;
      ok = true;
    }
  }
  return ok;
}

__device__ inline bool intersect(Ray r, IntersectResult& result) {
  bool ok = false;
  result.distance = FLT_MAX;
  ok = intersect(param.spheres, param.spheres_n, r, result) || ok;
  ok = intersect(param.planes, param.planes_n, r, result) || ok;
  return ok;
}

__device__ inline float3 sample(Ray r, int g_type, int g_id, float3 position,
                                float3 normal) {
  if (g_type)
    return param.planes[g_id].sample(r, position, normal);
  else 
    return param.spheres[g_id].sample(r, position, normal);
}

__device__ inline float3 gpuSample(Ray ray) {
  float3 color = make_float3(0, 0, 0);
  float reflectiveness = 1.0;
  float r = 1.0;
  float3 c = make_float3(0, 0, 0);
  IntersectResult ir;
  
  for (int i = 0; i < maxReflect + 1; ++i) {
    if (!intersect(ray, ir)) break;
    color += reflectiveness * (1 - r) * c;
    reflectiveness = reflectiveness * r;
    r = ir.reflectiveness;
    c = sample(ray, ir.g_type, ir.g_id, ir.position, ir.normal);
    if (r > 0) {
      Ray newray = {ir.position,
                 ir.normal * (-2*dot(ir.normal,ray.direction)) + ray.direction};
      ray = newray;
    } else
      break;
  }
  return color + reflectiveness * c;
}

__global__ void gpuRayTracing(unsigned* out) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  int y = index / width, x = index % width;
  float sx = x / float(width), sy = y / float(height);
  Ray r = param.camera.generateRay(sx, sy);
  float3 c = gpuSample(r);
  unsigned char c4[] = {
      __saturatef(c.z) * 255,
      __saturatef(c.y) * 255,
      __saturatef(c.x) * 255,
      255};

  unsigned ct = *reinterpret_cast<unsigned*>(c4); 
  out[index] = ct;
}

int main() {
  unsigned* gpuout;
  cudaSetDevice(0);
  cudaMallocHost(&image, width * height * sizeof(Color));
  cudaMalloc(&gpuout, sizeof(Color) * width * height);
  cudaMemcpyToSymbol(param, &cpuparam, sizeof RayTracingParam);
  clock_t t1 = clock(); 
  gpuRayTracing<<<width * height / 256, 256>>>(gpuout);
  cudaMemcpy(image, gpuout, sizeof(Color) * width * height, cudaMemcpyDeviceToHost); 
  clock_t t2 = clock();

  printf("%f\n", (t2 - t1) / float(CLOCKS_PER_SEC));
  cudaFree(gpuout);
  writebmp("raytracing_cuda.bmp", image, width, height);
  cudaFreeHost(image);
  return 0;
}
