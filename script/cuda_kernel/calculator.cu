// 입자 위치 갱신에 사용하는 구문
extern "C"{

    // linked list 초기화
    __global__ void init_head(int* head, int n_cells){

        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < n_cells){
            head[i] = -1; // 비어있는 상태
        }

    }

    __global__ void build_cell_list(const double* pos, int* head, int* next, const double* box, const int* grid_dim, const double* cell_size, const int N){

        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= N) return;

        double x = pos[3*i + 0];
        double y = pos[3*i + 1];
        double z = pos[3*i + 2];

        int ix = (int)(x / cell_size[0]);
        int iy = (int)(y / cell_size[1]);
        int iz = (int)(z / cell_size[2]);

        ix = max(0, min(ix, grid_dim[0] - 1));
        iy = max(0, min(iy, grid_dim[1] - 1));
        iz = max(0, min(iz, grid_dim[2] - 1));

        int cell_idx = ix + iy * grid_dim[0] + iz * grid_dim[0] * grid_dim[1];

        int old_head = atomicExch(&head[cell_idx], i);
        next[i] = old_head;

    }

    __global__ void update_p1(double* pos, double* vel, const double* force, const double* mass, const double dt, const double* box, const int N){
    
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= N) return;

        double dt_half = 0.5 * dt;
        double m = mass[i];

        // 1. 속도 절반 갱신
        double vx = vel[3*i + 0] + (force[3*i + 0] / m) * dt_half;
        double vy = vel[3*i + 1] + (force[3*i + 1] / m) * dt_half;
        double vz = vel[3*i + 2] + (force[3*i + 2] / m) * dt_half;

        // 2. 위치 갱신
        double rx = pos[3*i + 0] + vx * dt;
        double ry = pos[3*i + 1] + vy * dt;
        double rz = pos[3*i + 2] + vz * dt;

        // 3. 주기적 경계 조건 처리
        double Lx = box[0];
        double Ly = box[1];
        double Lz = box[2];

        // 공간을 벗어나면 반대로 넘김
        rx = rx - Lx * floor(rx / Lx);
        ry = ry - Ly * floor(ry / Ly);
        rz = rz - Lz * floor(rz / Lz);

        // 결과 저장
        pos[3*i + 0] = rx; pos[3*i + 1] = ry; pos[3*i + 2] = rz;
        vel[3*i + 0] = vx; vel[3*i + 1] = vy; vel[3*i + 2] = vz;
    }

    __global__ void compute_force(const double* pos, double* force, const int* head, const int* next, const double* box, const int* grid_dim, const double* cell_size, const double cutoff_sq, const int N){

        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= N) return;

        double xi = pos[3*i + 0];
        double yi = pos[3*i + 1];
        double zi = pos[3*i + 2];

        double fx = 0.0, fy = 0.0, fz = 0.0;
        
        // 공간의 절반 계산
        double Lx = box[0]; double Ly = box[1]; double Lz = box[2];
        double Lx_half = Lx * 0.5;
        double Ly_half = Ly * 0.5;
        double Lz_half = Lz * 0.5;
        
        int ix = (int)(xi / cell_size[0]);
        int iy = (int)(yi / cell_size[1]);
        int iz = (int)(zi / cell_size[2]);

        ix = max(0, min(ix, grid_dim[0] - 1));
        iy = max(0, min(iy, grid_dim[1] - 1));
        iz = max(0, min(iz, grid_dim[2] - 1));

        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {

                    int nx_c = (ix + dx + grid_dim[0]) % grid_dim[0];
                    int ny_c = (iy + dy + grid_dim[1]) % grid_dim[1];
                    int nz_c = (iz + dz + grid_dim[2]) % grid_dim[2];

                    int neighbor_cell = nx_c + ny_c * grid_dim[0] + nz_c * grid_dim[0] * grid_dim[1];

                    int j = head[neighbor_cell];

                    while (j != -1){
                        if (i != j){

                            double dx_dist = xi - pos[3*j + 0];
                            double dy_dist = yi - pos[3*j + 1];
                            double dz_dist = zi - pos[3*j + 2];

                            if (dx_dist > Lx_half) dx_dist -= Lx;
                            else if (dx_dist < -Lx_half) dx_dist += Lx;

                            if (dy_dist > Ly_half) dy_dist -= Ly;
                            else if (dy_dist < -Ly_half) dy_dist += Ly;

                            if (dz_dist > Lz_half) dz_dist -= Lz;
                            else if (dz_dist < -Lz_half) dz_dist += Lz;

                            double r2 = dx_dist*dx_dist + dy_dist*dy_dist + dz_dist*dz_dist;

                            if (r2 < cutoff_sq){
                                double inv_r2 = 1.0 / r2;
                                double inv_r6 = inv_r2 * inv_r2 * inv_r2;
                                double inv_r12 = inv_r6 * inv_r6;
                                double factor = 48.0 * inv_r2 * (inv_r12 - 0.5 * inv_r6);

                                fx += factor * dx_dist;
                                fy += factor * dy_dist;
                                fz += factor * dz_dist;
                            }
                        }
                        j = next[j]; // 격자 내 다음 입자에 대해 수행
                    }
                }
            }
        }
        force[3*i + 0] = fx; force[3*i + 1] = fy; force[3*i + 2] = fz;
    }

    __global__ void update_p2(double* vel, const double* force, const double* mass, const double dt, const int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= N) return;

        double dt_half = 0.5*dt;
        double m = mass[i];

        // v = v + 0.5 * a * dt
        vel[3*i + 0] += (force[3*i + 0] / m) * dt_half;
        vel[3*i + 1] += (force[3*i + 1] / m) * dt_half;
        vel[3*i + 2] += (force[3*i + 2] / m) * dt_half;
    }

} // extern "C"