(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects e j i)
(:init 
(harmony)
(planet e)
(planet j)
(planet i)
(province e)
(province j)
(province i)
)
(:goal
(and
(craves e j)
(craves j i)
)))