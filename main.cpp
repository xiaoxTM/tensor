//#include "u-random.hpp"
#include "u-tensor.hpp"
#include <iostream>
#include "u-shape.hpp"
#include <libu/u-timer>
#include <libu/u-checker>
#include <libu/u-log>

void tensor_test() {
    // Tensor construct function test
	u::tensor::Tensor t1;
	u_ncheck(t1.ref() == nullptr && t1.cref() == nullptr);
	u_ncheck (t1.type() == u::tensor::DType::invalid);
	u_ncheck (t1.rank() == 0);

	// Tensor(DType type) : data_(nullptr), type_(type) {}
	u::tensor::Tensor t3(u::tensor::DType::int32);
	u_ncheck (t3.type() == u::tensor::DType::int32);
	u_ncheck (t3.ref() == nullptr && t3.cref() == nullptr);
	u_ncheck (t1.rank() == 0);

	// Tensor(const Shape& shape, DType type, bool lazy = false) : shape_(shape), type_(type)
	u::tensor::Shape shape({3,5,9});
	u::tensor::Tensor t4(shape, u::tensor::DType::uint8);
	u_ncheck (t4.rank() == 3);
	u_ncheck (t4.ref() != nullptr);
	u_ncheck (t4.type() == u::tensor::DType::uint8);

    u::tensor::Shape shape1({8,5,10,4,9});
	u::tensor::Tensor t5(shape1, u::tensor::DType::uint16, true);
	u_ncheck (t5.rank() == 5);
	u_ncheck (t5.ref() == nullptr);
	u_ncheck (t5.type() == u::tensor::DType::uint16);

    //Tensor(unsigned char * data, const std::vector<size_t> &shape, DType type, bool copy = false)
	u::tensor::Tensor t6({3,5,9}, u::tensor::DType::uint8);
	u_ncheck (t6.rank() == 3);
	u_ncheck (t6.ref() != nullptr);
	u_ncheck (t6.type() == u::tensor::DType::uint8);

	u::tensor::Tensor t7({8,5,10,4,9}, u::tensor::DType::uint16, true);
	u_ncheck (t7.rank() == 5);
	u_ncheck (t7.ref() == nullptr);
	u_ncheck (t7.type() == u::tensor::DType::uint16);

	float data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
			17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
			34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 };
    float data2[] = {1,2,3,4,5,6,7,8,9,10,
                     11,12,13,14,15,16,17,18,19,20,
                     21,22,23,24,25,26,27,28,29,30,
                     31,32,33,34,35,36,37,38,39,40,
                     41,42,43,44,45,46,47,48};

    u::tensor::Tensor t8(reinterpret_cast<unsigned char *>(data), {2,3,8}, u::tensor::DType::float32);
    u::tensor::Tensor t9(reinterpret_cast<unsigned char *>(data2), {2,3,8}, u::tensor::DType::float32);
    std::cout << t8 + t9 << std::endl;
    std::cout << t8 + 10 << std::endl;
    std::cout << 10 + t8 << std::endl;
    std::cout << t8 - t9 << std::endl;
	std::cout << t9 - 10 << std::endl;
	std::cout << 10 - t9 << std::endl;
    std::cout << t8 * t9 << std::endl;
	std::cout << t8 * 10 << std::endl;
	std::cout << 10 * t8 << std::endl;
    std::cout << t8 / t9 << std::endl;
	std::cout << t9 / 10 << std::endl;
	std::cout << 10 / t9 << std::endl;
    std::cout << ++ t8 << std::endl;
	std::cout << t9 ++ << std::endl;
    u_ncheck(u::tensor::Tensor::all(t8 != t9));
    u_ncheck(u::tensor::Tensor::any(t8 != t9));
    u_ncheck(u::tensor::Tensor::any(t8+1 == t9));
    u_ncheck(u::tensor::Tensor::any(t8 == t9-1));

    u::tensor::Tensor t10 = t8.tile({1,2,2});
	u_ncheck(t10.rank() == 3);
	u_ncheck(t10.shape() == u::tensor::Shape({2,6,16}));

    std::cout << t8 << std::endl;
	std::cout << "--------------------" << std::endl;
	std::cout << t10 << std::endl;

	float data3[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    u::tensor::Tensor t11(reinterpret_cast<unsigned char *>(data3), {1,1,4,5}, u::tensor::DType::float32);
	std::cout << t11 << std::endl;
	std::cout << t11.broadcast(u::tensor::Shape({2,2,4,5})) << std::endl;
	std::cout << t11.tile({2,3,2,1}) << std::endl;
}

void shape_test() {
    u::tensor::Shape shape({3, 1, 5, 20});
	u_ncheck(shape==u::tensor::Shape({3,1,5,20}));
	u_ncheck(shape!=u::tensor::Shape({3,1,3,20}));
	u_ncheck(shape!=u::tensor::Shape({3,1,3,20}))
	u_ncheck(shape.size() == shape.rank());
	u_ncheck(shape.size() != 3);
	u_ncheck(shape.size() == 4);
    u_ncheck(shape.broadcast({3, 5, 1, 20}) == u::tensor::Shape({3, 5, 5, 20}));
	u_ncheck(shape.broadcastable(u::tensor::Shape({3,2,5,20})));
	u_ncheck(shape.broadcastable(u::tensor::Shape({1,3,1,20})));
	u_ncheck(!shape.broadcastable(u::tensor::Shape({2,1,5,20})));
	u_ncheck(!shape.broadcastable(u::tensor::Shape({3,1,4,20})));
    u_ncheck(!shape.broadcastable(u::tensor::Shape({2,1,5,20}), true));
    u_ncheck(shape.broadcastable(u::tensor::Shape({1,5,20}), true));
    u_ncheck(!shape.broadcastable(u::tensor::Shape({1,3,1,5,20}), true));
}

int main(int argc, char *argv[]) {
	//u::log::open(0xFF00 | u::D);
	u_fun_enter(0, 0);

    shape_test();

	tensor_test();

	u::checker::summary();
	u_fun_exit(0, 0);
	return (0);
}
