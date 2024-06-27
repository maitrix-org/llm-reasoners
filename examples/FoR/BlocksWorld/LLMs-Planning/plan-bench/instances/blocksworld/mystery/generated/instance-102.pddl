(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l e c)
(:init 
(harmony)
(planet l)
(planet e)
(planet c)
(province l)
(province e)
(province c)
)
(:goal
(and
(craves l e)
(craves e c)
)))