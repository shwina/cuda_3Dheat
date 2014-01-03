

void init_temp(float &M, int N_x, int N_y, N_z){

    int i, j, k;

    for (i=0; i<N_x; i++){
        for (j=0; j<N_y; j++){
            for (k=0; k<N_z; k++){

                i3d = k*(N_x*N_y) + j*(N_x) + i;

                if (i == 0 || j == 0 || k == 0){
                    M[i3d] = 10.0f;
                }

                if (i == N_x-1 || j == N_y-1 || j == N_z - 1){
                    M[i3d] = 10.0f;
                }

                else{
                    M[i3d] = 0.0f;
                }
        }
    }


}

