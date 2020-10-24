#include <nlohmann/json.hpp>
#include <fstream>
#include <mitie/ner_trainer.h>
#include <iostream>
#include <list>

//#define PATH_MITIE_MODEL "../MITIE-models/english/total_word_feature_extractor.dat"
//#define PATH_SAVE_NEW_MODEL "here.dat"
//#define DATASET_TRAIN "/home/alcides/pie_b2w/datasets/summer/parsed_329_modelo_summer.json"

using namespace dlib;
using namespace std;
using namespace mitie;

using json = nlohmann::json;

std::vector<string> split (const string &s, char delim) {
	std::vector<string> result;
	stringstream ss (s);
	string item;

	while (getline (ss, item, delim)) {
		result.push_back (item);
	}

	return result;
}

int main(int argc, char** argv) {
	string file("/home/alcides/pie_b2w/datasets/summer/parsed_329_modelo_summer.json");
	ifstream reader(file);
	json data;

	reader >> data;

	if (strcmp("/home/alcides/pie_b2w/datasets/summer/parsed_329_modelo_summer.json", "parsed_29_modelo_summer.json") == 0) {
		data = data[0]["trainings"];
	} else {
		data = data["trainings"];
	}

	list<ner_training_instance> list_training_obj;

	cout << data;
	for (json i : data) {
		std::vector<std::string> sentence;
		string text(i["text"]);
		string token("");
		for (int j = 0; j < text.size(); j++) {
			if (text[j] == ' ' ) {
				sentence.push_back(token);
				token = "";
			} else if (j == text.size() - 1) {
				token += text[j];
				sentence.push_back(token);
			} else {
				token += text[j];
			}
		}
		ner_training_instance training_obj(sentence);
		for (json tag : i["tags"]) {
			training_obj.add_entity((int) tag["start"], (int)tag["end"]  - (int) tag["start"], (new string(tag["tag"]))->c_str());
			cout << "Token start " << (int) tag["start"] << " size " << (int) tag["end"] - (int) tag["start"] << " label " << (new string(tag["tag"]))->c_str() << endl;
			for (int j = (int) tag["start"]; j < (int) tag["end"]; j++) {
				cout << sentence[j];
			}
			cout << endl;
		}
		list_training_obj.push_back(training_obj);
		cout << text << endl;
	}


	ner_trainer trainer("../MITIE-models/english/total_word_feature_extractor.dat");
	trainer.set_num_threads(4);

	for (auto &i : list_training_obj) {
		trainer.add(i);
	}

	named_entity_extractor ner = trainer.train();

    serialize("new_ner_model.dat") << "mitie::named_entity_extractor" << ner;

	const std::vector<string> tagstr = ner.get_tag_name_strings();
	cout << "The tagger supports " << tagstr.size() << " tags:" << endl;
	for (unsigned int i = 0; i < tagstr.size(); ++i)
		cout << "   " << tagstr[i] << endl;

	std::vector<std::string> sentence3;
	sentence3.push_back("Camisa");    
	sentence3.push_back("de");
	sentence3.push_back("manga");
	sentence3.push_back("listrada");
	sentence3.push_back("colorida");

	std::vector<pair<unsigned long, unsigned long> > chunks;
	std::vector<unsigned long> chunk_tags;
	ner(sentence3, chunks, chunk_tags);
	cout << "\nNumber of named entities detected: " << chunks.size() << endl;
	for (unsigned int i = 0; i < chunks.size(); ++i)
	{
		cout << "   Tag " << chunk_tags[i] << ":" << tagstr[chunk_tags[i]] << ": ";
		for (unsigned long j = chunks[i].first; j < chunks[i].second; ++j)
			cout << sentence3[j] << " ";
		cout << endl;
	}

	return 0;



}