(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g j f b)
(:init 
(harmony)
(planet g)
(planet j)
(planet f)
(planet b)
(province g)
(province j)
(province f)
(province b)
)
(:goal
(and
(craves g j)
(craves j f)
(craves f b)
)))