import numpy as np

class Hamming:
    '''
    Hamming (7,4) error correction code implementation.
    Can be used to encode, parity check, error correct, decode and get the orginal message back.
    This can detect two bit errors and correct single bit errors.
    '''
    _gT = np.matrix([[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0], [
        0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    _h = np.matrix(
        [[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]])

    _R = np.matrix([[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [
        0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])

    
    def _strToMat(self, binaryStr):
        '''
        @Input
        Binary string of length 4

        @Output
        np row vector of length 4
        '''

        inp = np.frombuffer(binaryStr.encode(), dtype=np.uint8)-ord('0')
        return inp

    
    def encode(self, message):
        '''
        @Input
        String
        Message is a 4 bit binary string
        @Output
        np matrix column vector
        Encoded 7 bit binary string
        '''
        message = np.matrix(self._strToMat(message)).transpose()
        en = np.dot(self._gT, message) % 2
        return en

    
    def parityCheck(self, message):
        '''
        @Input
        np matrix a column vector of length 7
        Accepts a binary column vector

        @Output
        np row vector of length 3
        Returns the single bit error location as row vector
        '''
        z = np.dot(self._h, message) % 2
        return np.fliplr(z.transpose())

    
    def getOriginalMessage(self, message):
        '''
        @Input
        np matrix a column vector of length 7
        Accepts a binary column vector

        @Output
        List of length 4
        Returns the single bit error location as row vector ()
        '''
        ep = self.parityCheck(message)
        pos = self._binatodeci(ep)
        if pos > 0:
            correctMessage = self._flipbit(message, pos)
        else:
            correctMessage = message

        origMessage = np.dot(self._R, correctMessage)
        return origMessage.transpose().tolist()

    
    def _flipbit(self, enc, bitpos):
        '''
        @Input
          enc:np matrix a column vector of length 7
          Accepts a binary column vector

          bitpos: Integer value of the position to change
          flip the bit. Value should be on range 1-7

        @Output
          np matrix a column vector of length 7
          Returns the bit flipped matrix
        '''
        enc = enc.transpose().tolist()
        bitpos = bitpos - 1
        if (enc[0][bitpos] == 1):
            enc[0][bitpos] = 0
        else:
                enc[0][bitpos] = 1
        return np.matrix(enc).transpose()

    
    def _binatodeci(self, binaryList):
        '''
        @Input
        np matrix column or row one dimension

        @Output
        Decimal number equal to the binary matrix
        '''
        return sum(val*(2**idx) for idx, val in enumerate(reversed(binaryList.tolist()[0])))


if __name__ == "__main__":
    ham = Hamming()
    mark = np.load("ammhackati.npy")
    mark = "".join([str(x) for x in mark])
    mark_str = [mark[i:i+4] for i in range(0, len(mark), 4)]
    enc = []
    for s in mark_str:
        enc.append(ham.encode(s))
    
    print(type(enc[0]))
    


    # print(enc)

    orig = []
    for x in enc:
        e = ham.getOriginalMessage(x)
        orig.append("".join([str(l) for l in e[0]]))
    
    orig = "".join(orig)
    print(orig == mark)