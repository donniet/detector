package detect

/*
#cgo LDFLAGS: -ldetector

#include <face_detector_wrapper.h>
#include <facenet_wrapper.h>
#include <multi_modal_lib.h>

typedef unsigned int uint;

#include <stdio.h>

float get_embedding_at(float * embedding, int i) {
	return embedding[i];
}

detection * get_response_detection(response * res, unsigned long dex) {
  if(dex < res->num_detections) {
    return res->detections + dex;
  }
  return NULL;
}

void deserialize_helper(multi_modal_wrapper * wrapper, void * input_buf, unsigned long input_size) {
	mm_deserialize(wrapper, (char*)input_buf, input_size);
}

distribution_wrapper * get_peak(distribution_wrapper * dist, uint i) {
	return &dist[i];
}
float get_element(float * arr, uint i) {
	return arr[i];
}
char get_byte(char * arr, unsigned long i) {
	return arr[i];
}

*/
import "C"

import (
	"fmt"
	"image"
	"image/color"
	"io"
	"io/ioutil"
	"log"
	"math"
	"unsafe"
)

const (
	sqrt2 = 1.41421356237
	eps   = 1e-6
)

/*
MultiModal is a structure to store a mixture of spherically symmetric multi-dimensional gaussian distributions
*/
type MultiModal struct {
	wrapper *C.multi_modal_wrapper
}

/*
Distribution is a struct that represents a single peak in the MultiModal structure
*/
type Distribution struct {
	Mean   []float32
	StdDev float32
	Count  int
	Id     uint64
}

/*
Erf is a version of the error function which operates on spherically symmetric gassian
distributions.  It's actually not the proper math for a multiple-dimension gaussian,
but hopefully good enough for now...
*/
func (d Distribution) Erf(vector []float32) (er float32, dis float32) {
	l := len(vector)
	if l > len(d.Mean) {
		l = len(d.Mean)
	}
	diff := make([]float32, l)

	for i := 0; i < l; i++ {
		diff[i] = vector[i] - d.Mean[i]
	}

	var acc float32
	for _, x := range diff {
		acc += x * x
	}

	y := math.Sqrt(float64(acc))
	log.Printf("dist: %f", y)

	if y < eps {
		return 0., float32(y)
	} else if d.StdDev < eps {
		return 1., float32(y)
	}

	return float32(math.Erf(y / float64(d.StdDev) / sqrt2)), float32(y)
}

/*
NewMultiModal creates a new MultiModal structure
*/
func NewMultiModal(dimensions int, maximumNodes int) MultiModal {
	return MultiModal{
		wrapper: C.mm_create(C.ulong(dimensions), C.ulong(maximumNodes)),
	}
}

/*
Dimensions returns the number of dimensions specified in the construction
*/
func (mm MultiModal) Dimensions() int {
	return int(C.mm_get_dimensions(mm.wrapper))
}

/*
WriteTo implements the WriterTo interface and is used to save the data structure to a writer
*/
func (mm MultiModal) WriteTo(w io.Writer) (n int64, err error) {
	var buf *C.char
	var siz C.ulong
	C.mm_serialize(mm.wrapper, &buf, &siz)
	defer C.mm_destroy_serialize_buffer(mm.wrapper, buf, siz)

	n = int64(siz)

	log.Printf("serializing %d bytes", n)

	b := make([]byte, n)
	for i := int64(0); i < n; i++ {
		b[i] = byte(C.get_byte(buf, C.ulong(i)))
	}

	if n, err := w.Write(b); err != nil {
		log.Fatalf("error writing mm: %d %v", n, err)
	}
	return
}

/*
ReadFrom implements the ReaderFrom interface and is used for restoring the MultiModal structure from a reader
*/
func (mm MultiModal) ReadFrom(r io.Reader) (int64, error) {
	b, err := ioutil.ReadAll(r)

	if err != nil {
		return int64(len(b)), err
	}

	C.deserialize_helper(mm.wrapper, unsafe.Pointer(&b[0]), C.ulong(len(b)))
	return int64(len(b)), nil
}

/*
Close must be called to clean up the memory used by MultiModal.  Failing to call this method before the multimodal leaves scope will result in a large memory leak.
*/
func (mm MultiModal) Close() {
	C.mm_destroy(mm.wrapper)
}

/*
Insert adds a float vector to the data structure
*/
func (mm MultiModal) Insert(vector []float32) {
	dat := make([]C.float, len(vector))
	for i, f := range vector {
		dat[i] = C.float(f)
	}
	C.mm_insert(mm.wrapper, &dat[0], C.ulong(len(vector)))
}

/*
Count returns the total number of vectors inserted into the data structure
*/
func (mm MultiModal) Count() int {
	return int(C.mm_get_count(mm.wrapper))
}

/*
Find returns the best distribution given the input float vector
*/
func (mm MultiModal) Find(vector []float32) Distribution {
	var dist *C.distribution_wrapper
	var count C.ulong

	dat := make([]C.float, len(vector))
	for i, f := range vector {
		dat[i] = C.float(f)
	}

	C.mm_find_peak(mm.wrapper, &dat[0], C.ulong(len(vector)), &dist, &count)
	defer C.mm_destroy_peaks(mm.wrapper, dist, count)

	d := C.get_peak(dist, C.uint(0))

	mean := make([]float32, mm.Dimensions())
	for j := 0; j < mm.Dimensions(); j++ {
		mean[j] = float32(C.get_element(d.mean, C.uint(j)))
	}

	return Distribution{
		Mean:   mean,
		StdDev: float32(d.standard_deviation),
		Count:  int(d.sample_count),
		Id:     uint64(d.id),
	}

}

/*
Peaks returns all the likely peaks in the data structure as Distributions
*/
func (mm MultiModal) Peaks() []Distribution {
	var dist *C.distribution_wrapper
	var count C.ulong
	var ret []Distribution

	C.mm_extract_peaks(mm.wrapper, &dist, &count)
	defer C.mm_destroy_peaks(mm.wrapper, dist, count)

	for i := 0; i < int(count); i++ {
		d := C.get_peak(dist, C.uint(i))

		mean := make([]float32, mm.Dimensions())
		for j := 0; j < mm.Dimensions(); j++ {
			mean[j] = float32(C.get_element(d.mean, C.uint(j)))
		}

		ret = append(ret, Distribution{
			Mean:   mean,
			StdDev: float32(d.standard_deviation),
			Count:  int(d.sample_count),
			Id:     uint64(d.id),
		})
	}

	return ret
}

/*
RGB24 is a structure holding a 24-bit raw RGB image
*/
type RGB24 struct {
	Pix    []uint8
	Stride int
	Rect   image.Rectangle
}

/*
RGB is a single pixel in the RGB24 image type
*/
type RGB struct {
	R, G, B uint8
}

/*
RGB24Reader wraps an io.Reader but returns RGB24 images of dimension given by Rect. This is used fo read raw pixels from a pipe or file.
*/
type RGB24Reader struct {
	Reader io.Reader
	Rect   image.Rectangle
}

/*
ReadRGB24 reads 3 * width * height bits and puts them into an RGB24 image in row major order
TODO: update to read in from multiple UDP clients and use from address to create seperate buffers
*/
func (r *RGB24Reader) ReadRGB24() (*RGB24, error) {
	buf := make([]byte, r.Rect.Dx()*r.Rect.Dy()*3)
	if len(buf) == 0 {
		return nil, fmt.Errorf("cannot read zero pixel image")
	}

	// log.Printf("expecting %d bytes per frame.", len(buf))

	cur := 0
	for {
		n, err := r.Reader.Read(buf[cur:])

		if err != nil {
			return nil, err
		} else if n+cur < len(buf) {
			cur += n
		} else {
			break
		}
	}

	return &RGB24{
		Pix:    buf,
		Stride: r.Rect.Dx() * 3,
		Rect:   r.Rect,
	}, nil
}

/*
RGBModel is the color model for a 24-bit image
*/
var RGBModel color.Model = color.ModelFunc(func(c color.Color) color.Color {
	r, g, b, _ := c.RGBA()
	return RGB{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8)}
})

/*
RGBA implements the Color interface for the RGB pixel type
*/
func (c RGB) RGBA() (r, g, b, a uint32) {
	r = uint32(c.R) << 8
	g = uint32(c.G) << 8
	b = uint32(c.B) << 8
	return
}

/*
NewRGB creates a black RGB24 image
*/
func NewRGB(r image.Rectangle) *RGB24 {
	return &RGB24{
		Rect:   r.Canon(),
		Stride: 3 * r.Dx(),
		Pix:    make([]uint8, 3*r.Dx()*r.Dy()),
	}
}

/*
FromImage constructs an RGB24 from a given image
*/
func FromImage(img image.Image) *RGB24 {
	if r, ok := img.(*RGB24); ok {
		return r
	}
	// this is really slow for now...
	r := NewRGB(img.Bounds())
	for x := r.Rect.Min.X; x < r.Rect.Max.X; x++ {
		for y := r.Rect.Min.Y; y < r.Rect.Max.Y; y++ {
			r.Set(x, y, img.At(x, y))
		}
	}
	return r
}

/*
FromRaw constructs an RGB24 image from raw bytes in RGB-row major order
*/
func FromRaw(b []byte, stride int) *RGB24 {
	return &RGB24{
		Pix:    b,
		Stride: stride,
		Rect:   image.Rect(0, 0, stride/3, len(b)/stride/3),
	}
}

/*
At implements the image.Image interface for RGB24
*/
func (p *RGB24) At(x, y int) color.Color {
	if !(image.Point{x, y}.In(p.Rect)) {
		return RGB{}
	}
	i := p.PixOffset(x, y)
	return RGB{
		p.Pix[i], p.Pix[i+1], p.Pix[i+2],
	}
}

/*
Set implements the image.Image interface for RGB24
*/
func (p *RGB24) Set(x, y int, c color.Color) {
	if !(image.Point{x, y}.In(p.Rect)) {
		return
	}
	i := p.PixOffset(x, y)
	c1 := RGBModel.Convert(c).(RGB)
	p.Pix[i+0] = uint8(c1.R)
	p.Pix[i+1] = uint8(c1.G)
	p.Pix[i+2] = uint8(c1.B)
}

/*
ColorModel implements the image.Image interface for RGB24
*/
func (p *RGB24) ColorModel() color.Model {
	return RGBModel
}

/*
SubImage returns an *RGB24 which is the crop of the given image using the rectangle r
*/
func (p *RGB24) SubImage(r image.Rectangle) image.Image {
	r = r.Intersect(p.Rect)
	// If r1 and r2 are Rectangles, r1.Intersect(r2) is not guaranteed to be inside
	// either r1 or r2 if the intersection is empty. Without explicitly checking for
	// this, the Pix[i:] expression below can panic.
	if r.Empty() {
		return &RGB24{}
	}
	// TODO: implement this much faster sub image routine, but this requires image stride
	//   in the C code.  right now just copy the image bytes to a new slice
	// i := p.PixOffset(r.Min.X, r.Min.Y)
	// return &RGB24{
	// 	Pix:    p.Pix[i:],
	// 	Stride: p.Stride,
	// 	Rect:   r,
	// }
	ret := &RGB24{
		Stride: r.Dx() * 3,
		Rect:   image.Rect(0, 0, r.Dx(), r.Dy()),
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			i := p.PixOffset(x, y)
			ret.Pix = append(ret.Pix, p.Pix[i], p.Pix[i+1], p.Pix[i+2])
		}
	}
	return ret
}

// PixOffset returns the index of the first element of Pix that corresponds to
// the pixel at (x, y).
func (p *RGB24) PixOffset(x, y int) int {
	return (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*3
}

/*
Bounds returns the bounding rectangle of the image
*/
func (p *RGB24) Bounds() image.Rectangle {
	return p.Rect
}

/*
Embedding is a float vector
*/
type Embedding []float32

/*
Classifier is a type used to get embeddings from images
*/
type Classifier struct {
	Description string
	Weights     string
	Device      string
	classer     *C.struct_Facenet
}

/*
ClassifierResponse is the response from a classification request
*/
type ClassifierResponse struct {
	Duration  float32
	Embedding Embedding
}

/*
NewClassifier takes an OpenVINO description and weights file and creates a classifier for the given device
*/
func NewClassifier(descriptionFile string, weightsFile string, device string) *Classifier {
	return &Classifier{
		Description: descriptionFile,
		Weights:     weightsFile,
		Device:      device,
		classer:     C.create_classifier(C.CString(descriptionFile), C.CString(weightsFile), C.CString(device)),
	}
}

/*
Close must be called to free memory from the Classifier object
*/
func (c *Classifier) Close() {
	C.destroy_classifier(c.classer)
}

/*
EmbeddingSize returns the number of dimensions of the output embedding
*/
func (c *Classifier) EmbeddingSize() int {
	return int(C.classifier_get_embedding_size(c.classer))
}

/*
InferRGB24 takes an RGB24 image and gets a ClassifierResponse
*/
func (c *Classifier) InferRGB24(rgb *RGB24) ClassifierResponse {
	res := C.classifier_do_classification(c.classer, unsafe.Pointer(&rgb.Pix[0]), C.int(rgb.Stride),
		C.int(rgb.Rect.Min.X), C.int(rgb.Rect.Min.Y), C.int(rgb.Rect.Max.X), C.int(rgb.Rect.Max.Y))
	// res := C.do_classification_param(c.classer, unsafe.Pointer(&rgb.Pix[0]), C.uint(rgb.Bounds().Dx()), C.uint(rgb.Bounds().Dy()))
	defer C.destroy_classifier_response(res)

	ret := ClassifierResponse{
		Duration: float32(res.duration),
	}

	for i := C.ulong(0); i < res.embedding_size; i++ {
		ret.Embedding = append(ret.Embedding, float32(C.get_embedding_at(res.embedding, C.int(i))))
	}

	return ret
}

/*
Detector wraps an OpenVINO detector network
*/
type Detector struct {
	Description string
	Weights     string
	Device      string
	detect      *C.struct_FaceDetector
}

/*
Detection is a single detection from a Detector object
*/
type Detection struct {
	Confidence float32
	Label      float32
	Rect       image.Rectangle
}

/*
NewDetector constructs a Detector from an OpenVINO description and weights file
*/
func NewDetector(descriptionFile string, weightsFile string, deviceName string) *Detector {
	ret := &Detector{
		Description: descriptionFile,
		Weights:     weightsFile,
		Device:      deviceName,
	}
	ret.detect = C.detector_create(
		C.CString(descriptionFile),
		C.CString(weightsFile),
		C.CString(deviceName))
	return ret
}

/*
Close cleans up the memory of the detector.  This must be called to ensure no memory leaks
*/
func (d *Detector) Close() {
	C.detector_destroy(d.detect)
	d.detect = nil
}

/*
InferRGB takes an RGB24 image and returns a slice of Detection objects
*/
func (d *Detector) InferRGB(rgb *RGB24) []Detection {
	res := C.detector_do_inference(d.detect, unsafe.Pointer(&rgb.Pix[0]),
		C.int(rgb.Stride), C.int(rgb.Rect.Min.X), C.int(rgb.Rect.Min.Y), C.int(rgb.Rect.Max.X), C.int(rgb.Rect.Max.Y))
	defer C.detector_destroy_response(res)

	var ret []Detection

	for i := C.ulong(0); i < res.num_detections; i++ {
		det := C.get_response_detection(res, i)

		x0 := math.Max(float64(det.xmin), 0)
		x1 := math.Min(float64(det.xmax), 1)
		y0 := math.Max(float64(det.ymin), 0)
		y1 := math.Min(float64(det.ymax), 1)

		tec := Detection{
			Confidence: float32(det.confidence),
			Label:      float32(det.label),
			Rect: image.Rect(
				int(x0*float64(rgb.Bounds().Dx()))+rgb.Bounds().Min.X,
				int(y0*float64(rgb.Bounds().Dy()))+rgb.Bounds().Min.Y,
				int(x1*float64(rgb.Bounds().Dx()))+rgb.Bounds().Min.X,
				int(y1*float64(rgb.Bounds().Dy()))+rgb.Bounds().Min.Y),
		}

		ret = append(ret, tec)
	}
	return ret
}
