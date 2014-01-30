# include <iostream>
# include <iomanip>
# include <fstream>


template <typename T>
void write_to_file(T* ary, int N, char* S){

    char buf[100];
    strcpy(buf, "");
    strcat(buf, S);
    std::ofstream output(buf);
    for(int i=0; i<N; i++){
        output << ary[i] << ", ";
    }
}