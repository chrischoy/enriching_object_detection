__global__ void scrambleGammaToSigma( float* Sigma, float* Gamma, float lambda, int* nonEmptyRows, int* nonEmptyCols, int GammaDim, int HOGDim, int nNonEmptyCells )
{
    int r = blockDim.x * blockIdx.x + threadIdx.x; // rows
    int c = blockDim.y * blockIdx.y + threadIdx.y; // cols 

    int sigmaDim = HOGDim * nNonEmptyCells;
    if( r < sigmaDim && c < sigmaDim ){
        int HOG_row_idx = r % HOGDim;
        int HOG_col_idx = c % HOGDim;

        int currCellIdx = r / HOGDim;
        int otherCellIdx = c / HOGDim;

        int gammaRowIdx = abs( nonEmptyRows[currCellIdx] - nonEmptyRows[otherCellIdx] );
        int gammaColIdx = abs( nonEmptyCols[currCellIdx] - nonEmptyCols[otherCellIdx] );
        Sigma[r + c * sigmaDim] = Gamma[ ((gammaRowIdx * HOGDim) + HOG_row_idx) + ( ( gammaColIdx * HOGDim ) + HOG_col_idx ) * GammaDim ]; // + (r==c)?lambda:0 ;
        if (r == c) Sigma[r + c * sigmaDim] += lambda;
    }
}