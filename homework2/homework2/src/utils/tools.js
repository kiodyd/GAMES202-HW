function getRotationPrecomputeL(precompute_L, rotationMatrix) {
    let rotMat = mat4Matrix2mathMatrix(rotationMatrix)
    let rotateSHMat1 = computeSquareMatrix_3by3(rotMat)
    let rotateSHMat2 = computeSquareMatrix_5by5(rotMat)
    let ret = []

    for (i = 0; i < 3; i++) {
        let colors = math.clone(getMat3ValueFromRGB(precompute_L)[i])

        let sh1n1 = colors[1]
        let sh10 = colors[2]
        let sh1p1 = colors[3]

        let sh2n2 = colors[4]
        let sh2n1 = colors[5]
        let sh20 = colors[6]
        let sh2p1 = colors[7]
        let sh2p2 = colors[8]

        let rotatedSH1 = math.multiply(rotateSHMat1, [sh1n1, sh10, sh1p1])._data
        let rotatedSH2 = math.multiply(rotateSHMat2, [sh2n2, sh2n1, sh20, sh2p1, sh2p2])._data

        colors[1] = rotatedSH1[0]
        colors[2] = rotatedSH1[1]
        colors[3] = rotatedSH1[2]

        colors[4] = rotatedSH2[0]
        colors[5] = rotatedSH2[1]
        colors[6] = rotatedSH2[2]
        colors[7] = rotatedSH2[3]
        colors[8] = rotatedSH2[4]

        ret.push([colors[0], colors[1], colors[2],
        colors[3], colors[4], colors[5],
        colors[6], colors[7], colors[8]])
    }
    return ret;
}

function computeSquareMatrix_3by3(rotationMatrix) { // 计算方阵SA(-1) 3*3 

    // 1、pick ni - {ni}
    let n1 = [1, 0, 0, 0]; let n2 = [0, 0, 1, 0]; let n3 = [0, 1, 0, 0];

    // 2、{P(ni)} - A  A_inverse
    let Pn1 = SHEval(n1[0], n1[1], n1[2], 3);
    let Pn2 = SHEval(n2[0], n2[1], n2[2], 3);
    let Pn3 = SHEval(n3[0], n3[1], n3[2], 3);

    // 取对应层的投影值
    let A = math.matrix([
        [Pn1[1], Pn1[2], Pn1[3]],
        [Pn2[1], Pn2[2], Pn2[3]],
        [Pn3[1], Pn3[2], Pn3[3]]]);
    let A_inverse = math.inv(A);

    // 3、用 R 旋转 ni - {R(ni)}
    let Rn1 = math.multiply(rotationMatrix, n1)._data;
    let Rn2 = math.multiply(rotationMatrix, n2)._data;
    let Rn3 = math.multiply(rotationMatrix, n3)._data;

    // 4、R(ni) SH投影 - S
    let P_Rn1 = SHEval(Rn1[0], Rn1[1], Rn1[2], 3);
    let P_Rn2 = SHEval(Rn2[0], Rn2[1], Rn2[2], 3);
    let P_Rn3 = SHEval(Rn3[0], Rn3[1], Rn3[2], 3);
    let S = math.matrix([
        [P_Rn1[1], P_Rn1[2], P_Rn1[3]],
        [P_Rn2[1], P_Rn2[2], P_Rn2[3]],
        [P_Rn3[1], P_Rn3[2], P_Rn3[3]]]);

    // 5、S*A_inverse
    return math.multiply(S, A_inverse);
}

function computeSquareMatrix_5by5(rotationMatrix) { // 计算方阵SA(-1) 5*5

    // 1、pick ni - {ni}
    let k = 1 / math.sqrt(2);
    let n1 = [1, 0, 0, 0]; let n2 = [0, 0, 1, 0]; let n3 = [k, k, 0, 0];
    let n4 = [k, 0, k, 0]; let n5 = [0, k, k, 0];

    // 2、{P(ni)} - A  A_inverse
    let Pn1 = SHEval(n1[0], n1[1], n1[2], 3);
    let Pn2 = SHEval(n2[0], n2[1], n2[2], 3);
    let Pn3 = SHEval(n3[0], n3[1], n3[2], 3);
    let Pn4 = SHEval(n4[0], n4[1], n4[2], 3);
    let Pn5 = SHEval(n5[0], n5[1], n5[2], 3);

    // 取对应层的投影值
    let A = math.matrix([
        [Pn1[4], Pn1[5], Pn1[6], Pn1[7], Pn1[8]],
        [Pn2[4], Pn2[5], Pn2[6], Pn2[7], Pn2[8]],
        [Pn3[4], Pn3[5], Pn3[6], Pn3[7], Pn3[8]],
        [Pn4[4], Pn4[5], Pn4[6], Pn4[7], Pn4[8]],
        [Pn5[4], Pn5[5], Pn5[6], Pn5[7], Pn5[8]]]);
    let A_inverse = math.inv(A);

    // 3、用 R 旋转 ni - {R(ni)}
    let Rn1 = math.multiply(rotationMatrix, n1)._data;
    let Rn2 = math.multiply(rotationMatrix, n2)._data;
    let Rn3 = math.multiply(rotationMatrix, n3)._data;
    let Rn4 = math.multiply(rotationMatrix, n4)._data;
    let Rn5 = math.multiply(rotationMatrix, n5)._data;

    // 4、R(ni) SH投影 - S
    let P_Rn1 = SHEval(Rn1[0], Rn1[1], Rn1[2], 3);
    let P_Rn2 = SHEval(Rn2[0], Rn2[1], Rn2[2], 3);
    let P_Rn3 = SHEval(Rn3[0], Rn3[1], Rn3[2], 3);
    let P_Rn4 = SHEval(Rn4[0], Rn4[1], Rn4[2], 3);
    let P_Rn5 = SHEval(Rn5[0], Rn5[1], Rn5[2], 3);
    let S = math.matrix([
        [P_Rn1[4], P_Rn1[5], P_Rn1[6], P_Rn1[7], P_Rn1[8]],
        [P_Rn2[4], P_Rn2[5], P_Rn2[6], P_Rn2[7], P_Rn2[8]],
        [P_Rn3[4], P_Rn3[5], P_Rn3[6], P_Rn3[7], P_Rn3[8]],
        [P_Rn4[4], P_Rn4[5], P_Rn4[6], P_Rn4[7], P_Rn4[8]],
        [P_Rn5[4], P_Rn5[5], P_Rn5[6], P_Rn5[7], P_Rn5[8]]]);

    // 5、S*A_inverse
    return math.multiply(S, A_inverse);
}

function mat4Matrix2mathMatrix(rotationMatrix) {

    let mathMatrix = [];
    for (let i = 0; i < 4; i++) {
        let r = [];
        for (let j = 0; j < 4; j++) {
            r.push(rotationMatrix[i * 4 + j]);
        }
        mathMatrix.push(r);
    }
    return math.matrix(mathMatrix)

}

function getMat3ValueFromRGB(precomputeL) {

    let colorMat3 = [];
    for (var i = 0; i < 3; i++) {
        colorMat3[i] = mat3.fromValues(precomputeL[0][i], precomputeL[1][i], precomputeL[2][i],
            precomputeL[3][i], precomputeL[4][i], precomputeL[5][i],
            precomputeL[6][i], precomputeL[7][i], precomputeL[8][i]);
    }
    return colorMat3;
}