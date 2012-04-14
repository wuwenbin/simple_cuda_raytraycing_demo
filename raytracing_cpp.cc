#include <fstream>
#include <cstring>
#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include <cfloat>
#include <ctime>
#include <cstdint>
using namespace std;
const int width = 2048, height = 2048;
const int maxReflect = 5;
inline float clampf(float v, float min, float max) {
  return v < min ? min : v > max ? max : v;
}

struct Color {
  float r, g, b;
  Color(float r_ = 0.0, float g_ = 0.0, float b_ = 0.0) :
      r(r_), g(g_), b(b_){}
  Color(const Color& c) : r(c.r), g(c.g), b(c.b) {}
  Color clamp() const {
    return Color( clampf(r, 0.0, 1.0),
                  clampf(g, 0.0, 1.0),
                  clampf(b, 0.0, 1.0));
  }

  Color operator+(const Color& o) const{
    return Color(r + o.r , g + o.g, b + o.b);
  }

  Color& operator+=(const Color& o) {
    r += o.r; g += o.g; b += o.b;
    return *this;
  }
  Color operator*(float v) const {
    return Color(v * r, v * g, v * b);
  }

  Color modulate(const Color& o) const {
    return Color(r * o.r, g * o.g, b * o.b);
  }
};
inline Color operator*(float v, const Color& o) {return o * v;}
Color image[width * height];
class BMPWriter {
 public:
  struct bmpheader {
    uint32_t filesz;
    uint16_t creator1;
    uint16_t creator2;
    uint32_t bmp_offset;
  };
  
  struct bmpinfo{
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
  };

  struct bgracolor {
    unsigned char b,g,r,a;
  };

  static void write_header(ofstream& f, int width, int height) {
    bmpheader header;
    bmpinfo info;
    int size = width * height * 4;
    memset(&header, 0, sizeof(header));
    memset(&info, 0, sizeof(info));
    header.filesz = size + 54;
    header.bmp_offset = 54;
    info.header_sz = 40;
    info.width = width;
    info.height = height;
    info.nplanes = 1;
    info.bitspp = 32;
    info.bmp_bytesz = size;
    f.write("BM", 2);
    f.write((char*)&header, sizeof(header));
    f.write((char*)&info, sizeof(info));
  }

  static void conv_color(bgracolor* out, const Color* in, int size) {
    for (int i = 0; i < size; ++i) {
      Color c = in[i].clamp();
      out[i].r = c.r * 255;
      out[i].g = c.g * 255;
      out[i].b = c.b * 255;
      out[i].a = 255;
    }
  }

  static void write_color(ofstream& f, bgracolor* bgra, int width, int height) {
    f.write((char*)bgra, width * height * sizeof(*bgra));
  }
  static void write(const std::string& filename, const Color* data, int width, int height) {
 
    ofstream f(filename.c_str(), ios::binary);
    if (!f) return;
   
    write_header(f, width, height);
    
    bgracolor* bgra = new bgracolor[width * height];
    conv_color(bgra, data, width * height);
    write_color(f, bgra, width, height);
   
    delete[] bgra;
  }
};

struct Vector {
  float x, y, z;
  Vector(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
  Vector(const Vector& r) : x(r.x), y(r.y), z(r.z) {}
  float sqrLength() const {
    return x * x + y * y + z * z;
  }
  
  float length() const {
    return sqrt(sqrLength());
  }

  Vector operator+(const Vector& r) const {
    return Vector(x + r.x, y + r.y, z + r.z);
  }

  Vector operator-(const Vector& r) const {
    return Vector(x - r.x, y - r.y, z - r.z);
  }

  Vector operator*(float v) const {
    return Vector(v * x, v * y, v * z);
  }

  Vector operator/(float v) const {
    float inv = 1 / v;
    return *this * inv;
  }

  Vector normalize() const {
    float invlen = 1 / length();
    return *this * invlen;
  }

  float dot(const Vector& r) const {
    return x * r.x + y * r.y + z * r.z;
  }

  Vector cross(const Vector& r) const {
    return Vector(-z * r.y + y * r.z,
                  z * r.x - x * r.z,
                  -y * r.x + x * r.y);
  }

  static Vector zero() {
    return Vector(0, 0, 0);
  }
};
inline Vector operator*(float l, const Vector& r) {return r * l;}

struct Ray {
  Vector origin, direction;
  Ray(const Vector& o, const Vector& d) : origin(o), direction(d) {}

  Vector getPoint(float t) const {
    return origin + t * direction;
  }
};

class Geometry;

struct IntersectResult {
  const Geometry* geometry;
  float distance;
  Vector position;
  Vector normal;

  IntersectResult() : geometry(NULL), distance(0), position(Vector::zero()), normal(Vector::zero()) {}
  IntersectResult(const Geometry* g, float d, const Vector& p, const Vector& n) :
      geometry(g), distance(d), position(p), normal(n) {}
};

struct Scene {
  virtual ~Scene() {}
  virtual IntersectResult intersect(const Ray& ray) const = 0;
};

struct Material {
  float reflectiveness;
  virtual ~Material() {}
  Material(float r = 0.0) : reflectiveness(r) {}
  virtual Color sample(const Ray& ray, const Vector& position, const Vector& normal) const = 0;
};

struct Geometry : public Scene {
  virtual ~Geometry() {}
  unique_ptr<Material> material;
  Geometry(Material* m = NULL) : material(m) {}
};
struct Sphere : public Geometry {
  Vector center;
  float radius, sqrRadius;

  Sphere(const Vector& c, float r, Material* m = NULL) :
      Geometry(m), center(c), radius(r), sqrRadius(r * r) {}
  IntersectResult intersect(const Ray& ray) const {
    Vector v = ray.origin - center;
    float a0 = v.sqrLength() - sqrRadius;
    float DdotV = ray.direction.dot(v);
    if (DdotV <= 0.0) {
      float discr = DdotV * DdotV - a0;
      if (discr >= 0.0) {
        float d = -DdotV - sqrt(discr);
        Vector p = ray.getPoint(d);
        Vector n = (p - center).normalize();
        return IntersectResult(this, d, p, n);
      }
    }
    return IntersectResult();
  }
};

struct Plane : public Geometry {
  Vector normal, position;
  std::auto_ptr<Material> material;

  Plane(const Vector& n, float d, Material* m = NULL) :
      Geometry(m), normal(n), position(normal * d) {}

  IntersectResult intersect(const Ray& ray) const {
    float a = ray.direction.dot(normal);
    if (a >= 0.0)
      return IntersectResult();
    float b = normal.dot(ray.origin - position);
    float d = -b / a;
    return IntersectResult(this, d, ray.getPoint(d), normal);
  }
};

struct Union : public Scene {
  unique_ptr<Geometry> geometries[4];
  int n;
  /*
  template <typename... T>
  Union(T... gs) {
    addGeometries(gs...);
  }

  void addGeometries() {
    
  }
  template <typename... T>
  void addGeometries(Geometry* g, T... gs) {
    geometries.push_back(shared_ptr<Geometry>(g));
    addGeometries(gs...);
  }
  */
  Union(Geometry* g1 = NULL, Geometry* g2 = NULL, Geometry* g3 = NULL, Geometry* g4 = NULL) {
    n = 0;
    if (g1) geometries[n++].reset(g1);
    if (g2) geometries[n++].reset(g2);
    if (g3) geometries[n++].reset(g3);
    if (g4) geometries[n++].reset(g4);
  }
  IntersectResult intersect(const Ray& ray) const {
    
    float minDistance = FLT_MAX;
    IntersectResult minResult;
    for (int i = 0; i <n; ++i) {
      IntersectResult result = geometries[i]->intersect(ray);
      if (result.geometry && result.distance < minDistance) {
        minDistance = result.distance;
        minResult = result;
      }
    }
    return minResult;
  }
};

struct PerspectiveCamera {
  Vector eye, front, right, up;
  float fovScale;
  PerspectiveCamera(const Vector& e, const Vector& f, const Vector& u, float fov)
      :eye(e), front(f), right(f.cross(u)), up(right.cross(f)), fovScale(tan(fov * 0.5 * 3.1415926 / 180) * 2) {}

  Ray generateRay(float x, float y) const {
    Vector r = right * ((x - 0.5) * fovScale);
    Vector u = up * ((y - 0.5) * fovScale);
    return Ray(eye, (front + r + u).normalize());
  }
};

inline int iabs(int x) {
  return x < 0 ? -x : x;
}
struct CheckerMaterial : public Material {
  float scale;
  CheckerMaterial(float s, float r = 0.0) : Material(r), scale(s) {}
  Color sample(const Ray& ray, const Vector& position, const Vector& normal) const {
    if (iabs(floor(position.x * 0.1) + floor(position.z * scale)) % 2 < 1)
      return Color();
    else
      return Color(1.0, 1.0, 1.0);
  }
};
Vector lightDir = Vector(1, 1, 1).normalize();
Color lightColor = Color(1, 1, 1);
inline float max(float a, float b) {
  return a > b ? a : b;
}
struct PhongMaterial : public Material {
  Color diffuse, specular;
  int shininess;
  PhongMaterial(const Color& d, const Color& sp, int sh, float r = 0.0) :
      Material(r), diffuse(d), specular(sp), shininess(sh) {}

  Color sample(const Ray& ray, const Vector& position, const Vector& normal) const {
    float NdotL = normal.dot(lightDir);
    Vector H = (lightDir - ray.direction).normalize();
    float NdotH = normal.dot(H);
    Color diffuseTerm = diffuse * max(NdotL, 0.0);
    Color specularTerm = specular * pow(max(NdotH, 0.0), shininess);
    //Color r = lightColor.modulate(diffuseTerm + specularTerm);
    //cerr << r.r << " " << r.g << " " << r.b << endl;
    return lightColor.modulate(diffuseTerm + specularTerm);
  }
};

template <int maxReflect>
struct RayTracer {
  Color operator()(const Scene& scene, const Ray& ray_) const {
    Color color;
    float reflectiveness = 1.0;
    int reflect_times = 0;
    float r = 1.0;
    Color c;
    Ray ray = ray_;
    for (int i = 0; i < maxReflect; ++i) {
      IntersectResult result;
      result = scene.intersect(ray);
      if (!result.geometry) {
        reflect_times = i;
        break;
      }
      color += reflectiveness * (1 - r) * c;
      reflectiveness = reflectiveness * r;
      r = result.geometry->material->reflectiveness;
      c = result.geometry->material->sample(ray, result.position, result.normal);
      if (reflectiveness > 0) {
        ray = Ray(result.position, result.normal * (-2 * result.normal.dot(ray.direction)) + ray.direction);
      } else
        break;
    }
    return color + reflectiveness * c;
  }
};

template <typename RenderFuncT>
void render(const Scene& scene, const PerspectiveCamera& camera, const RenderFuncT& f) {
#pragma omp parallel for num_threads(4)
  for (int i = 0; i < height * width; ++i) {
    int y = i / width;
    int x = i % width;
    float sy = y / float(height);
    float sx = x / float(width);
    Ray ray = camera.generateRay(sx, sy);
    image[i] = f(scene, ray);
  }
}

int main() {
  clock_t t1 = clock();
  render(Union(new Plane(Vector(0, 1, 0), 0, new CheckerMaterial(0.1, 0.25)),
               new Sphere(Vector(-10, 10, -10), 10, new PhongMaterial(Color(1, 0, 0), Color(1, 1, 1),16, 0.25)),
               new Sphere(Vector( 10, 10, -10), 10, new PhongMaterial(Color(0, 0, 1), Color(1, 1, 1), 16, 0.25))),
         PerspectiveCamera(Vector(0, 5, 15), Vector(0, 0, -1), Vector(0, 1, 0), 90),
         RayTracer<maxReflect>());
  clock_t t2 = clock();
  cerr << (t2 - t1) / float(CLOCKS_PER_SEC) << endl;
  BMPWriter::write("raytracing.bmp", image, width, height);
}
