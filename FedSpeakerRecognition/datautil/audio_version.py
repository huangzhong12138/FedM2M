import glob

from pydub import AudioSegment
import os


def voxceleb2_data(file_address):
    print("voxceleb2_data_to_text方法成功被调用")
    flac_files = glob.glob(os.path.join(file_address, "*", "*", "*.m4a"))
    spk_to_utts = dict()
    speakers = []  # 唯一说话人标记的列表，为之后分组客户端用
    for flac_file in flac_files:
        split_name = flac_file.split('\\')
        # print(split_name[1])
        spk = split_name[1]
        if spk not in spk_to_utts:
            spk_to_utts[spk] = [flac_file.replace('\\', '/')]
            speakers.append(spk)
        else:
            spk_to_utts[spk].append(flac_file.replace('\\', '/'))
    # args.num_classes = len(speakers)
    # print(spk_to_utts)
    # print(speakers)
    # print(len(speakers))
    return spk_to_utts, speakers


def trans_flac_to_wav(file_path):
    file_dir = os.path.dirname(file_path)
    new_name = os.path.basename(file_path).replace('.m4a', '.wav')
    new_file = os.path.join(file_dir, new_name)
    song = AudioSegment.from_file(file_path)
    song.export(new_file, format="wav")
    print(f'Converted {file_path} to {new_file}')
    return new_file


print('===== Begin to Do converter =====')
# audio_files,speakers = voxceleb2_data()
# for audio_file in audio_files:
#     # do converter
#     trans_flac_to_wav(audio_file)

if __name__ == '__main__':
    audio_files, speakers = voxceleb2_data("D:/AR_data/vox2_test_aac/aac")
    print(audio_files)
    # for speakers in audio_files:
    #     for audio_file in audio_files[speakers]:
    #         # do converter
    #         # print(audio_file)
    #         trans_flac_to_wav(audio_file)
    # 删除.m4a文件
    for speakers in audio_files:
        for audio_file in audio_files[speakers]:
            try:
                os.remove(audio_file)
                print(f'Successfully deleted {audio_file}')
            except FileNotFoundError:
                print(f'File {audio_file} not found')
            except Exception as e:
                print(f'Error deleting file {audio_file}: {e}')