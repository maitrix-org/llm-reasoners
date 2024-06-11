(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l e j)
(:init 
(harmony)
(planet l)
(planet e)
(planet j)
(province l)
(province e)
(province j)
)
(:goal
(and
(craves l e)
(craves e j)
)))