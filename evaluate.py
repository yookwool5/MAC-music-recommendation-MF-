# -*- coding: utf-8 -*-
import fire
import numpy as np

from arena_util import load_json


class ArenaEvaluator:
    def _idcg(self, l):
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self):
        self._idcgs = [self._idcg(i) for i in range(101)]

    def _ndcg(self, gt, rec, k):
        dcg = 0.0
        for i, r in enumerate(rec):
            if r in gt:
                dcg += 1.0 / np.log(i + 2)
            if i == k-1 :
                break

        return dcg / self._idcgs[len(gt)]
    
    def mAP(self, gt, rec, k):
        score = []
        suc = 0
        for i, r in enumerate(rec):
            user_map = []
            if r in gt:
                suc += 1
                a = suc / (i + 1)
                user_map.append(a)
            if user_map:  
                score.append(sum(user_map) / len(user_map))
            if i == k-1 :
                break
        mAP = sum(score) / len(score) if score else 0 
        return mAP


    def _eval(self, gt_playlists, rec_playlists, k):        
        gt_dict = {g["id"]: g for g in gt_playlists}

        gt_ids = set([g["id"] for g in gt_playlists])
        rec_ids = set([r["id"] for r in rec_playlists])

        if gt_ids != rec_ids:
            print(len(gt_ids), len(rec_ids))
            raise Exception("결과의 플레이리스트 수가 올바르지 않습니다.")

        rec_song_counts = [len(p["songs"]) for p in rec_playlists]

        if set(rec_song_counts) != set([k]):
            raise Exception("추천 곡 결과의 개수가 맞지 않습니다.")

        rec_unique_song_counts = [len(set(p["songs"])) for p in rec_playlists]

        if set(rec_unique_song_counts) != set([k]):
            raise Exception("한 플레이리스트에 중복된 곡 추천은 허용되지 않습니다.")

        music_ndcg = 0.0

        for rec in rec_playlists:
            gt = gt_dict[rec["id"]]
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:k])

        music_ndcg = music_ndcg / len(rec_playlists)
        
        music_mAP = 0
        
        for rec in rec_playlists:
            gt = gt_dict[rec["id"]]
            music_mAP += self.mAP(gt["songs"], rec["songs"][:k])
                                                   
        music_mAP = music_mAP / len(rec_playlists)                                    

        return music_ndcg, music_mAP

    def evaluate(self, gt_playlists, rec_playlists, k):
        try:
            music_ndcg, mAP = self._eval(gt_playlists, rec_playlists, k)
            print(f"nDCG: {music_ndcg:.6}")
            print(f"mAP: {mAP:.6}")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    fire.Fire(ArenaEvaluator)
