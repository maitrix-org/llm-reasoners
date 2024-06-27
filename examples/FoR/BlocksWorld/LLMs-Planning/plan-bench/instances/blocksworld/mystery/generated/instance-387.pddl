(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects b e f)
(:init 
(harmony)
(planet b)
(planet e)
(planet f)
(province b)
(province e)
(province f)
)
(:goal
(and
(craves b e)
(craves e f)
)))