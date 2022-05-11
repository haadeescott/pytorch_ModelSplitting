import os, random, struct, binascii
from Crypto.Cipher import AES
import time

def getPass(path):
    with open(path) as f:
        pwd = f.readlines()
        aesSecretDecryption = pwd[0]
        f.close()

        return aesSecretDecryption

def decrypt_file(key, in_filename, out_filename, chunksize=24*1024):
    """ Decrypts a file using AES (CBC mode) with the
        given key. Parameters are similar to encrypt_file,
        with one difference: out_filename, if not supplied
        will be in_filename without its last extension
        (i.e. if in_filename is 'aaa.zip.enc' then
        out_filename will be 'aaa.zip')
    """

    with open(in_filename, 'rb') as infile:
        origsize = struct.unpack('<Q', infile.read(struct.calcsize('Q')))[0]
        iv = infile.read(16)
        decryptor = AES.new(key.encode("utf8"), AES.MODE_CBC, iv)

        with open(out_filename, 'wb') as outfile:
            while True:
                chunk = infile.read(chunksize)
                if len(chunk) == 0:
                    break
                outfile.write(decryptor.decrypt(chunk))

            outfile.truncate(origsize)

start_time = time.time()

decrypt_file(getPass('secrets/secret_1.txt'), 'submodel_1_enc.pt', 'submodel_1.pt')
decrypt_file(getPass('secrets/secret_2.txt'), 'submodel_2_enc.pt', 'submodel_2.pt')
decrypt_file(getPass('secrets/secret_3.txt'), 'submodel_3_enc.pt', 'submodel_3.pt')
decrypt_file(getPass('secrets/secret_4.txt'), 'submodel_4_enc.pt', 'submodel_4.pt')
decrypt_file(getPass('secrets/secret_5.txt'), 'submodel_5_enc.pt', 'submodel_5.pt')
decrypt_file(getPass('secrets/secret_6.txt'), 'submodel_6_enc.pt', 'submodel_6.pt')
decrypt_file(getPass('secrets/secret_7.txt'), 'submodel_7_enc.pt', 'submodel_7.pt')
decrypt_file(getPass('secrets/secret_8.txt'), 'submodel_8_enc.pt', 'submodel_8.pt')
decrypt_file(getPass('secrets/secret_9.txt'), 'submodel_9_enc.pt', 'submodel_9.pt')
decrypt_file(getPass('secrets/secret_10.txt'), 'submodel_10_enc.pt', 'submodel_10.pt')
decrypt_file(getPass('secrets/secret_11.txt'), 'submodel_11_enc.pt', 'submodel_11.pt')
decrypt_file(getPass('secrets/secret_12.txt'), 'Main_Submodel2_enc.pt', 'Main_Submodel2.pt')

dc = time.time()

print("---File has been decrypted!---\n")

print("Total time taken for Decryption: {} secs".format(dc-start_time))