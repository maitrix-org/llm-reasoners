

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a c)
(ontable b)
(on c b)
(ontable d)
(clear a)
(clear d)
)
(:goal
(and
(on b c)
(on c d)
(on d a))
)
)


