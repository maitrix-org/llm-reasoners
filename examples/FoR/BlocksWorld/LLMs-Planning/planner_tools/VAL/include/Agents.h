#ifndef __AGENTS
#define __AGENTS

#include "ptree.h"
#include "Environment.h"

class Agents {
private:
	int numGps;
	std::vector<vector<VAL::const_symbol *> > agentGps;
	std::map<VAL::const_symbol *,int> inGroup;
	std::map<int,vector<pair<VAL::operator_ *,int> > > groupActions;
public:
	Agents() : numGps(0), agentGps() {};
	void addAgent(VAL::const_symbol * c)
	{
		agentGps[numGps-1].push_back(c);
		inGroup[c] = numGps-1;
	}
	void startNewGroup(VAL::const_symbol * c)
	{
		agentGps.push_back(std::vector<VAL::const_symbol *>());
		++numGps;
		addAgent(c);
	}

	void addAction(VAL::operator_ * o,int n)
	{
		groupActions[numGps-1].push_back(make_pair(o,n));
	}
	
	vector<int> whichGroups(const VAL::Environment & bdgs)
	{
		vector<int> gps;
		int c = 0;
		for(vector<vector<VAL::const_symbol *> >::iterator i = agentGps.begin();i != agentGps.end();++i,++c)
		{
			for(vector<VAL::const_symbol *>::iterator j = i->begin();j != i->end();++j)
			{
				for(VAL::Environment::const_iterator k = bdgs.begin();k != bdgs.end();++k)
				{
					if(k->second == *j)
					{
						gps.push_back(c);
						j = i->end();
						--j;
						break;
					}
				}
			}
		}
		return gps;
	}

	string show(int g)
	{
		string s("");
		for(vector<VAL::const_symbol *>::iterator i = agentGps[g].begin();
				i != agentGps[g].end();++i)
		{
			s += (*i)->getName();
			s += " ";
		}
		return s;
	}
};



#endif
