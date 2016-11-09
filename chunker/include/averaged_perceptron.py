from include.feature_vector import FeatureVector
from copy import deepcopy


class Learner():
    def __init__(self, feat_vec=None):
        if feat_vec:
            self.feat_vec = deepcopy(feat_vec)
        else:
            self.feat_vec = FeatureVector()
        return

    def train(self, dataList, f_argmax, f_gold_vec, iterations=1):
        feat_vec = self.feat_vec
        feat_vec_sum = FeatureVector()
        last_change_dict = FeatureVector()
        total_data_count = 0

        for iteration in range(iterations):
            data_total = len(dataList)
            data_count = 0

            for data in dataList:
                total_data_count += 1
                data_count += 1
                print "iteration", iteration, "data", data_count, "of", data_total

                local_vec = f_argmax(feat_vec, data)
                gold_vec = f_gold_vec(data)

                delta_vec = gold_vec - local_vec

                if data_count != data_total:
                    if not gold_vec == local_vec:
                        for key in delta_vec:
                            feat_vec_sum[key] += feat_vec[key] * (total_data_count - last_change_dict[key])
                            last_change_dict[key] = total_data_count

                        feat_vec += delta_vec
                        # Because feat_vec is updated here by adding delta_vec, we
                        # do exactly the same thing to feat_vec_sum, because it is
                        # in its nature, a sum of feat_vecs
                        feat_vec_sum += delta_vec
                else:
                    # Is the last data of current iteration, we need to apply
                    # all pending updates to feat_vec_sum
                    for key in last_change_dict.keys() + feat_vec.keys():
                        # Just to make sure we have indeed updated every key.
                        feat_vec_sum[key] += feat_vec[key] * (total_data_count - last_change_dict[key])
                        last_change_dict[key] = total_data_count

                    if not gold_vec == local_vec:
                        # Last but not least, don't forget the current delta_vec
                        feat_vec += delta_vec
                        feat_vec_sum += delta_vec

            # Dump every iteration
            tmp = feat_vec_sum / total_data_count
            tmp.dump("models/jetic_avg_Iter_" + str(iteration+1) + ".model")

        # Finalisation, divide feat_vec_sum with total_data_count
        self.feat_vec = feat_vec_sum / total_data_count

        return self.feat_vec.export()
