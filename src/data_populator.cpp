#include <iostream>
#include <fstream>
#include <cstdlib>   
#include <ctime>     
#include <sstream>   

int main(int argc, char* argv[]) 
{
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <number of random numbers> <optional:<output file name>>\n";
        return 1;
    }

    int n = 0;
    std::istringstream iss(argv[1]);
    if (!(iss >> n) || n <= 0) {
        std::cerr << "invalid number of random numbers: " << argv[1] << "\n";
        return 1;
    }


    std::srand(static_cast<unsigned int>(std::time(0)));

    std::ofstream outfile("data/generated_data.txt");
    if (!outfile) {
        std::cerr << "error: Could not open output file.\n";
        return 1;
    }

    outfile << n << " ";

    for (int i = 0; i < n; ++i) {
        int random_number = std::rand() % 100; 
        outfile << random_number << " ";  
    }

    outfile.close();  
    std::cout << "successfully wrote " << n << " random numbers to 'generated_data.txt'\n";

    return 0;
}