__kernel void MatrixMultiplicationTwo(__global float *matrix1,
				      __global float *matrix2,
				      __global float *matrix3,
				      __global float *temp,
				      __global float *temp2,
				      __global float *output, 
				      uint widthA, uint widthB)
{
	uint idy = get_global_id(0);
	uint idx = get_global_id(1);

	float sum = 0;

	for (int i=0;i<widthA;i++)
	{
		float tempRow1 = matrix1[idy * widthA + i];
		float tempRow2 = matrix2[i * widthB + idx];
		sum += tempRow1 * tempRow2;

	}

	temp[idy * widthA + idx] = sum;
	sum = 0;

	for (int i=0;i<widthA;i++)
	{
		float tempRow1 = temp[idy * widthA + i];
		float tempRow2 = matrix3[i * widthB + idx];
		sum += tempRow1 * tempRow2;

	}

	output[idy * widthA + idx] = sum;

	// Transpose matrix
	// uint targetIndex = idy * widthB + idx;
	// uint sourceIndex = idx * widthB + idy;
	// output[targetIndex] = temp2[sourceIndex];
}
