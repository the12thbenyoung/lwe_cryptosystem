import secrets
import numpy as np
import math
class LWECryptosystem():
    def __init__(self, n, l, m, q, r, t, alpha):
        self.n = n
        self.l = l
        self.m = m
        self.q = q
        self.r = r
        self.t = t
        self.alpha = alpha

        #private key
        self.S = np.array([[secrets.randbelow(q) for _ in range(l)] for _ in range(n)])

        #public key
        self.A = np.array([[secrets.randbelow(q) for _ in range(n)] for _ in range(m)])
        E = np.array([[self.sample_psi() for _ in range(l)]
                      for _ in range(m)])
        self.P = np.matmul(self.A, self.S) + E

    def encrypt(self, v: np.array):
        """encrypt v: l-length vector mod t to output (u,c) where u and c are n and l length
        vectors, respectively, mod q"""
        rand_range = list(range(-self.r, self.r+1))
        a = np.array([secrets.choice(rand_range) for _ in range(self.m)])
        return (np.matmul(np.transpose(self.A), a),
                np.matmul(np.transpose(self.P), a) + self.f(v))

    def decrypt(self, u: np.array, c: np.array):
        """u: length-n vector mod q
           c: length-l vector mod q (u and c are the two outputs of encrypt)
           return: length-l vector mod t (the original message)"""
        return self.f_inv(c - np.matmul(np.transpose(self.S), u))

    def sample_psi(self):
        """sample from normal distribution with mean 0 and std. dev. alpha*q/sqrt(2pi),
        round to nearest integer and reduce mod q"""
        return int(np.round(np.random.normal(0, self.alpha*(self.q)/math.sqrt(2*math.pi)))) % self.q

    def f(self, v: np.array):
        """v: l-length vector mod q
        output: l-length vector mod t obtained by multiplying each coordinate of v by
        q/t and rounding to the nearest integer"""
        return np.round(v*(self.q/self.t)).astype('int') % self.q

    def f_inv(self, v: np.array):
        """v: l-length vector mod t
        output: l-length vector mod q obtained by multiplying each coordinate of v by
        t/q and rounding to the nearest integer"""
        return np.round(v*(self.t/self.q)).astype('int') % self.t


def send_ciphertext(message: str, system: LWECryptosystem, hamming_encoder: np.array):
    n = system.n
    segment_bits = n//2 + 1
    #each character has 7-bit ascii code
    segment_chars = segment_bits // 7

    message_segments = [message[segment_chars*i:segment_chars*(i+1)]
                        for i in range(0, len(message)//segment_chars+1)]
    ciphertexts = []
    for segment in message_segments:
        plaintext = []
        #convert to binary representation of ascii codes
        for char in segment:
            ascii_binary = '{0:b}'.format(ord(char))
            #pad front with zeros to get to length 7
            ascii_binary = '0'*(7-len(ascii_binary)) + ascii_binary
            plaintext.extend(list(ascii_binary))
        for _ in range(len(segment), segment_chars):
            #pad with spaces if number of characters is below segment_chars
            plaintext.extend(list('0100000'))
        plaintext = np.array(plaintext).astype('int')
        #add bits to end of plaintext with hammond encoder
        encoded_plaintext = np.matmul(plaintext, hamming_encoder).astype('int') % 2

        #encrypt hammond-encoded bits with cryptosystem
        u, c = system.encrypt(encoded_plaintext)
        ciphertexts.append((u,c))

    return ciphertexts


def decrypt_ciphertext(ciphertexts: list, system: LWECryptosystem, hamming_decoder: np.array):
    n = system.n
    segment_bits = n//2 + 1
    #each character has 7-bit ascii code
    segment_chars = segment_bits // 7

    decrypted_message = ''
    for text in ciphertexts:
        u, c = text
        decrypted_text = system.decrypt(u, c)

        decoded_text = np.matmul(decrypted_text, hamming_decoder) % 2
        decoded_sum = sum(decoded_text)
        if decoded_sum == segment_bits-1:
            #first row of hamming_decoder is all 1s, so there was an error in the first bit
            decrypted_text[0] = (decrypted_text[0] + 1) % 2
        if sum(decoded_text) == segment_bits-2:
            #one of the next (segment_bits-1) message bits
            #error in the bit of wherever the 0 is plus 1
            error_location = np.argmin(decoded_text) + 1
            decrypted_text[error_location] = (decrypted_text[error_location] + 1) % 2
        #otherwise there's no error or error is in added hammond bits, so we don't care
        corrected_plaintext = decrypted_text[:segment_bits]

        binary_message = ''
        for bit in corrected_plaintext:
            binary_message += str(bit)

        #strip off bits 7 at a time and convert back from ascii to characters
        for i in range(0, len(binary_message), 7):
            decrypted_message += chr(int(binary_message[i:i+7], 2))

    return decrypted_message


def main():
    # characters per message segment
    segment_chars = 5
    # bits per message segment (each char encoded as 7-bit ascii)
    segment_bits = segment_chars * 7
    n = segment_bits * 2 - 1
    l = n
    m = 2008
    q = 2003
    r = 1
    t = 2
    alpha = 0.0065
    system = LWECryptosystem(n, l, m, q, r, t, alpha)

    #rounding procedure occasionally produces an error, so use a hamming code
    hamming_M = np.ones([1,segment_bits-1])
    for i in range(segment_bits-1):
        row = np.ones([1,segment_bits-1])
        row[0,i] = 0
        hamming_M = np.append(hamming_M, row, axis=0)

    hamming_encoder = np.append(np.identity(segment_bits), hamming_M, axis=1)
    hamming_decoder = np.append(hamming_M, np.identity(segment_bits-1), axis=0)

    message = 'learning with errors cryptosystem'
    print(message)
    ciphertexts = send_ciphertext(message, system, hamming_encoder)
    # print(ciphertexts)
    received_message = decrypt_ciphertext(ciphertexts, system, hamming_decoder)
    print(received_message)


if __name__ == '__main__':
    main()
