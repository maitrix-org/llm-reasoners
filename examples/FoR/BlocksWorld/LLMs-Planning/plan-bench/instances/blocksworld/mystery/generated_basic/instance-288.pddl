

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(planet b)
(craves c b)
(planet d)
(province a)
(province c)
(province d)
)
(:goal
(and
(craves b a)
(craves c d)
(craves d b))
)
)


