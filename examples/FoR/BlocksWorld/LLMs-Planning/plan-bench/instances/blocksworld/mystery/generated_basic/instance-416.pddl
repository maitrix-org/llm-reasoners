

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a b)
(craves b c)
(craves c d)
(planet d)
(province a)
)
(:goal
(and
(craves a c)
(craves c b))
)
)


