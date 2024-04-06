/**
 *  Example code using sampling to find KNN.
 *
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include "io.h"
#include "annoylib.h"
#include "kissrandom.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
// using namespace std;

float compare_with_id(const std::vector<float> &a, const std::vector<float> &b)
{
    float sum = 0.0;
    // Skip the first 2 dimensions
    for (size_t i = 2; i < a.size(); ++i)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

int main(int argc, char **argv)
{
    string source_path = "dummy-data.bin";
    string query_path = "dummy-queries.bin";
    string knn_save_path = "output.bin";

    // Also accept other path for source data
    if (argc > 1)
    {
        source_path = string(argv[1]);
    }

    uint32_t num_data_dimensions = 102;
    float sample_proportion = 0.001;

    // Read data points
    vector<vector<float>> nodes;
    ReadBin(source_path, num_data_dimensions, nodes);
    // cout<<nodes.size()<<"\n";
    // Read queries
    uint32_t num_query_dimensions = num_data_dimensions + 2;
    vector<vector<float>> queries;
    ReadBin(query_path, num_query_dimensions, queries);

    vector<vector<uint32_t>> knn_results; // for saving knn results

    uint32_t n = nodes.size();
    uint32_t d = nodes[0].size();
    uint32_t nq = queries.size();
    uint32_t sn = uint32_t(sample_proportion * n); // you don't need it right now
    uint32_t NUM_EXTRA_ATTRIBUTES = 2;             // for sample data input for building the annoy index

    // cout<<"# data points:  " << n<<"\n";
    // cout<<"# data point dim:  " << d<<"\n";
    // cout<<"# queries:      " << nq<<"\n";

    /** A basic method to compute the KNN results using sampling  **/
    const int K = 100; // To find 100-NN

    Annoy::AnnoyIndex<int, float, Annoy::Euclidean, Annoy::Kiss64Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy> annoyIndex(d);
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        annoyIndex.add_item(i, nodes[i].data() + NUM_EXTRA_ATTRIBUTES);
    }
    annoyIndex.build(10);

    NUM_EXTRA_ATTRIBUTES += 2;
    for (size_t i = 0; i < queries.size(); ++i)
    {
        std::vector<int> result_ids;
        std::vector<int> final_result_ids;
        if (queries[i][0] == 0)
        {
            // Vector-only query
            annoyIndex.get_nns_by_vector(queries[i].data() + NUM_EXTRA_ATTRIBUTES, 100, -1, &final_result_ids, nullptr);
        }
        else if (queries[i][0] == 1)
        {

            float categorical_value = static_cast<float>(queries[i][1]);
            annoyIndex.get_nns_by_vector(queries[i].data() + NUM_EXTRA_ATTRIBUTES, 200, -1, &result_ids, nullptr);
            for (int &idx : result_ids)
            {
                if (nodes[idx][0] == categorical_value)
                    final_result_ids.push_back(idx);
                if (final_result_ids.size() == 100)
                    break;
            }
        }
        else if (queries[i][0] == 2)
        {
            float timestamp_lower_bound = queries[i][2];
            float timestamp_upper_bound = queries[i][3];
            annoyIndex.get_nns_by_vector(queries[i].data() + NUM_EXTRA_ATTRIBUTES, 200, -1, &result_ids, nullptr);
            for (int &idx : result_ids)
            {
                if (timestamp_lower_bound <= nodes[idx][1] && nodes[idx][1] <= timestamp_upper_bound)
                    final_result_ids.push_back(idx);
                if (final_result_ids.size() == 100)
                    break;
            }
        }
        else if (queries[i][0] == 3)
        {
            int categorical_value = static_cast<int>(queries[i][1]);
            float timestamp_lower_bound = queries[i][2];
            float timestamp_upper_bound = queries[i][3];
            annoyIndex.get_nns_by_vector(queries[i].data() + NUM_EXTRA_ATTRIBUTES, 200, -1, &result_ids, nullptr);
            for (int &idx : result_ids)
            {
                if (timestamp_lower_bound <= nodes[idx][1] && nodes[idx][1] <= timestamp_upper_bound && nodes[idx][0] == categorical_value)
                    final_result_ids.push_back(idx);
                if (final_result_ids.size() == 100)
                    break;
            }
        }
        vector<uint32_t> converted_vec(final_result_ids.begin(), final_result_ids.end());

        // If the number of knn in the sampled data is less than K, then fill the rest with the last few nodes
        if (converted_vec.size() < K)
        {
            uint32_t s = 1;
            while (converted_vec.size() < K)
            {
                converted_vec.push_back(n - s);
                s = s + 1;
            }
        }

        knn_results.push_back(converted_vec);
    }

    SaveKNN(knn_results, knn_save_path);
    return 0;
}