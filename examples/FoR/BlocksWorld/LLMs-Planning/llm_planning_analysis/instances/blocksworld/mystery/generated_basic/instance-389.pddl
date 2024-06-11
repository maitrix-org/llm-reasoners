

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a b)
(craves b c)
(planet c)
(planet d)
(province a)
(province d)
)
(:goal
(and
(craves b a)
(craves c b)
(craves d c))
)
)


