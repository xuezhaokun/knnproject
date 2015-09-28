package p2;

import java.util.Comparator;

public class DistanceComparator implements Comparator<KnnData> {

	@Override
	public int compare(KnnData o1, KnnData o2) {
        if (o1.getDistance() < o2.getDistance())
        {
            return -1;
        }
        if (o1.getDistance() > o2.getDistance())
        {
            return 1;
        }
        return 0;
	}

}
