

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a d)
(ontable b)
(on c b)
(ontable d)
(clear a)
(clear c)
)
(:goal
(and
(on a c)
(on b a))
)
)


